"""
Kitsune Brain — LLM interface via LiteLLM.
Handles all communication with AI models, including fallback and streaming.
"""

import logging
import asyncio
import json
import queue
import re
import time
from dataclasses import dataclass
from typing import AsyncIterator

import litellm
from ollama import Client

from kitsune.config import Config
from kitsune.model_utils import pick_fast_model

logger = logging.getLogger("kitsune.brain")

# Suppress litellm's verbose logging
litellm.suppress_debug_info = True


@dataclass
class BrainResponse:
    """Response from the brain."""
    content: str
    model_used: str
    response_time: float
    tokens_used: int
    cost_estimate: float
    success: bool
    error: str | None = None
    error_type: str | None = None  # rate_limit, timeout, context_length, auth_failed, content_policy, unknown


class Brain:
    """Unified LLM interface with fallback and memory-augmented prompts."""

    def __init__(self, config: Config):
        self.config = config
        self._total_tokens = 0
        self._total_cost = 0.0
        logger.info("🧠 Brain initialized with providers: %s", config.available_providers)

    async def think(
        self,
        user_message: str,
        model: str,
        fallback_model: str | list[str],
        memory_context: str = "",
        conversation_history: list[dict] | None = None,
        user_name: str | None = None,
    ) -> BrainResponse:
        """
        Generate a response using the specified model with memory context.
        Auto-retries with backoff, then falls back through the chain.
        """
        messages = self._build_messages(user_message, memory_context, conversation_history, user_name)

        # Build candidate chain: primary + fallbacks + local Ollama safety net
        candidates = self._model_candidates(model, fallback_model)
        # If no Ollama candidate present but Ollama is configured, append a fast local model
        if self.config.ollama_api_key and not any(c.startswith("ollama/") for c in candidates if c):
            fast_local = getattr(self.config, "ollama_default_model", "ministral-3:3b-cloud")
            candidates.append(f"ollama/{fast_local}")

        last_error: Exception | None = None
        for i, current_model in enumerate(candidates):
            if not current_model:
                continue

            logger.info(
                "🤔 Thinking with %s%s...",
                current_model,
                " (fallback)" if i > 0 else "",
            )

            start_time = time.time()
            try:
                result = await self._invoke_with_retry(current_model, messages, stream=False)
                content, tokens, cost = result  # type: ignore[misc]
                elapsed = round(time.time() - start_time, 2)
                self._total_tokens += tokens
                self._total_cost += cost

                logger.info(
                    "✅ Response from %s in %.1fs (%d tokens, $%.4f)",
                    current_model,
                    elapsed,
                    tokens,
                    cost,
                )

                return BrainResponse(
                    content=content,
                    model_used=current_model,
                    response_time=elapsed,
                    tokens_used=tokens,
                    cost_estimate=cost,
                    success=True,
                )

            except Exception as e:
                last_error = e
                error_type = self._classify_error(e)
                logger.error(
                    "❌ Model %s exhausted retries: [%s] %s",
                    current_model,
                    error_type,
                    str(e)[:200],
                )
                if i == 0:
                    logger.info("↩️ Switching to fallback chain: %s", ", ".join(candidates[1:]) or "-")
                continue

        # All candidates failed
        final_error = str(last_error)[:200] if last_error else "All models failed"
        error_type = self._classify_error(last_error) if last_error else "unknown"
        return BrainResponse(
            content="Maaf, saya sedang mengalami masalah teknis. Coba lagi nanti ya 🙏",
            model_used="none",
            response_time=0,
            tokens_used=0,
            cost_estimate=0,
            success=False,
            error=final_error,
            error_type=error_type,
        )

    async def stream_think(
        self,
        user_message: str,
        model: str,
        fallback_model: str | list[str],
        memory_context: str = "",
        conversation_history: list[dict] | None = None,
        user_name: str | None = None,
    ) -> AsyncIterator[dict]:
        """Stream response chunks, then emit a final BrainResponse event."""
        messages = self._build_messages(user_message, memory_context, conversation_history, user_name)

        # Build candidate chain: primary + fallbacks + local Ollama safety net
        candidates = self._model_candidates(model, fallback_model)
        if self.config.ollama_api_key and not any(c.startswith("ollama/") for c in candidates if c):
            fast_local = getattr(self.config, "ollama_default_model", "ministral-3:3b-cloud")
            candidates.append(f"ollama/{fast_local}")

        last_error: Exception | None = None

        for i, current_model in enumerate(candidates):
            if not current_model:
                continue

            start_time = time.time()
            content_parts: list[str] = []
            tokens = 0
            cost = 0.0

            logger.info(
                "🌊 Streaming with %s%s...",
                current_model,
                " (fallback)" if i > 0 else "",
            )

            try:
                stream_iter = await self._invoke_with_retry(current_model, messages, stream=True)
                # stream_iter is an AsyncIterator[dict]
                async for chunk in stream_iter:  # type: ignore[union-attr]
                    if chunk.get("type") == "delta":
                        text = chunk.get("content", "")
                        content_parts.append(text)
                        yield chunk
                    elif chunk.get("type") == "usage":
                        tokens = int(chunk.get("tokens", 0))
                        cost = float(chunk.get("cost", 0.0))

                elapsed = round(time.time() - start_time, 2)
                content = "".join(content_parts)
                if not tokens:
                    tokens = len(content.split())
                self._total_tokens += tokens
                self._total_cost += cost
                yield {
                    "type": "final",
                    "response": BrainResponse(
                        content=content,
                        model_used=current_model,
                        response_time=elapsed,
                        tokens_used=tokens,
                        cost_estimate=cost,
                        success=True,
                    ),
                }
                return

            except Exception as e:
                last_error = e
                error_type = self._classify_error(e)
                logger.error(
                    "❌ Streaming model %s exhausted retries: [%s] %s",
                    current_model,
                    error_type,
                    str(e)[:200],
                )
                if i == 0:
                    logger.info("↩️ Switching streaming fallback chain to: %s", ", ".join(candidates[1:]) or "-")
                continue

        final_error = str(last_error)[:200] if last_error else "All streaming models failed"
        error_type = self._classify_error(last_error) if last_error else "unknown"
        yield {
            "type": "final",
            "response": BrainResponse(
                content="Maaf, saya sedang mengalami masalah teknis. Coba lagi nanti ya 🙏",
                model_used="none",
                response_time=0,
                tokens_used=0,
                cost_estimate=0,
                success=False,
                error=final_error,
                error_type=error_type,
            ),
        }

    def _build_messages(
        self,
        user_message: str,
        memory_context: str = "",
        conversation_history: list[dict] | None = None,
        user_name: str | None = None,
    ) -> list[dict]:
        system_prompt = self.config.prompt_templates.get(
            "system_prompt",
            "You are Kitsune 🦊, a helpful AI assistant.",
        )

        if user_name:
            system_prompt = f"{system_prompt}\n\nUser's name: {user_name}. Address them by name when appropriate."

        if memory_context:
            system_prompt = f"{system_prompt}\n\n{memory_context}"

        messages = [{"role": "system", "content": system_prompt}]
        if conversation_history:
            messages.extend(conversation_history[-6:])
        messages.append({"role": "user", "content": user_message})
        return messages

    @staticmethod
    def _model_candidates(model: str, fallback_model: str | list[str]) -> list[str]:
        raw_candidates = [model]
        if isinstance(fallback_model, list):
            raw_candidates.extend(fallback_model)
        elif fallback_model:
            raw_candidates.append(fallback_model)

        candidates = []
        seen = set()
        for candidate in raw_candidates:
            if not candidate or candidate in seen:
                continue
            candidates.append(candidate)
            seen.add(candidate)
        return candidates

    @staticmethod
    def _classify_error(error: Exception) -> str:
        """Classify an LLM error into a known category."""
        msg = str(error).lower()
        exc_type = type(error).__name__.lower()

        if any(k in msg for k in ("rate limit", "ratelimit", "too many requests", "429", "throttled")):
            return "rate_limit"
        if any(k in msg for k in ("timeout", "timed out", "connection reset", "eof", "broken pipe")):
            return "timeout"
        if any(k in msg for k in ("context length", "maximum context length", "token limit", "too large", "context_length_exceeded")):
            return "context_length"
        if any(k in msg for k in ("authentication", "auth", "unauthorized", "401", "403", "invalid api key", "incorrect api key")):
            return "auth_failed"
        if any(k in msg for k in ("content policy", "moderation", "blocked", "safety", "refusal", "responsibleai")):
            return "content_policy"
        if any(k in exc_type for k in ("connectionerror", "connecterror", "ssl", "certificate")):
            return "timeout"
        return "unknown"

    async def _invoke_with_retry(
        self,
        model: str,
        messages: list[dict],
        stream: bool = False,
    ) -> tuple[str, int, float] | AsyncIterator[dict]:
        """
        Invoke a model with exponential backoff retries.
        Returns (content, tokens, cost) for non-stream, or AsyncIterator for stream.
        """
        retries = 2
        delays = [1.0, 3.0]  # Exponential-ish backoff

        last_error: Exception | None = None

        for attempt in range(retries + 1):
            try:
                if model.startswith("ollama/"):
                    if stream:
                        return self._ollama_stream(model.removeprefix("ollama/"), messages)
                    return await self._ollama_chat(
                        model=model.removeprefix("ollama/"),
                        messages=messages,
                    )
                else:
                    if stream:
                        return self._litellm_stream(model, messages)
                    response = await litellm.acompletion(
                        model=model,
                        messages=messages,
                        max_tokens=4096,
                        temperature=0.7,
                    )
                    content = response.choices[0].message.content or ""
                    tokens = response.usage.total_tokens if response.usage else 0
                    cost = response._hidden_params.get("response_cost", 0.0) if hasattr(response, '_hidden_params') else 0.0
                    return content, tokens, cost
            except Exception as e:
                last_error = e
                error_type = self._classify_error(e)
                logger.warning(
                    "⚠️ Model %s failed (attempt %d/%d): [%s] %s",
                    model,
                    attempt + 1,
                    retries + 1,
                    error_type,
                    str(e)[:180],
                )
                if attempt < retries:
                    await asyncio.sleep(delays[attempt])

        raise last_error or RuntimeError(f"All retries exhausted for {model}")

    async def extract_memories(
        self,
        user_message: str,
        bot_response: str,
    ) -> list[dict]:
        """
        Use a fast model to extract notable facts/preferences from an interaction.
        Returns list of {fact, type} dicts.
        """
        extraction_prompt = self.config.prompt_templates.get(
            "memory_extraction_prompt",
            (
                "Extract only durable facts, preferences, corrections, or skills worth remembering "
                "for future conversations with this user. Do not store one-off requests. "
                "Return valid JSON only: a list of objects with fact, type, and topic fields. "
                "type must be one of preference, fact, correction, skill. "
                "If nothing durable is present, return [].\n\n"
                "User: {user_message}\nAssistant: {assistant_response}"
            ),
        )

        prompt = extraction_prompt.format(
            user_message=user_message[:500],
            assistant_response=bot_response[:500],
        )

        try:
            # Use the cheapest available model
            classifier_model = pick_fast_model(self.config)

            if classifier_model.startswith("ollama/"):
                content, _, _ = await self._ollama_chat(
                    model=classifier_model.removeprefix("ollama/"),
                    messages=[{"role": "user", "content": prompt}],
                )
                content = content.strip()
            else:
                response = await litellm.acompletion(
                    model=classifier_model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=500,
                    temperature=0.0,
                )
                content = response.choices[0].message.content.strip()

            # Parse JSON response, tolerating occasional markdown wrappers.
            if "```" in content:
                parts = content.split("```")
                content = parts[1] if len(parts) > 1 else content
                content = re.sub(r"^\s*json\s*", "", content, flags=re.IGNORECASE)
                content = content.strip()
            else:
                match = re.search(r"\[[\s\S]*\]", content)
                if match:
                    content = match.group(0)

            memories = json.loads(content)
            if isinstance(memories, list):
                return [
                    m for m in memories
                    if (
                        isinstance(m, dict)
                        and isinstance(m.get("fact"), str)
                        and m.get("type") in {"preference", "fact", "correction", "skill"}
                    )
                ]
            return []

        except Exception as e:
            logger.debug("Memory extraction failed (non-critical): %s", e)
            return []

    async def _ollama_chat(self, model: str, messages: list[dict]) -> tuple[str, int, float]:
        """Call Ollama's hosted API. The official client appends /api internally."""
        if not self.config.ollama_api_key:
            raise ValueError("OLLAMA_API_KEY is required for ollama/* models")

        def _call() -> tuple[str, int, float]:
            client = Client(
                host=self.config.ollama_api_base,
                headers={"Authorization": f"Bearer {self.config.ollama_api_key}"},
            )
            response = client.chat(
                model=model,
                messages=messages,
                options={"temperature": 0.7, "num_predict": 4096},
            )
            message = response.get("message", {}) if hasattr(response, "get") else getattr(response, "message", {})
            if isinstance(message, dict):
                content = message.get("content", "")
            else:
                content = getattr(message, "content", "")
            prompt_tokens = int((response.get("prompt_eval_count") if hasattr(response, "get") else getattr(response, "prompt_eval_count", 0)) or 0)
            output_tokens = int((response.get("eval_count") if hasattr(response, "get") else getattr(response, "eval_count", 0)) or 0)
            return content, prompt_tokens + output_tokens, 0.0

        return await asyncio.to_thread(_call)

    async def _litellm_stream(self, model: str, messages: list[dict]) -> AsyncIterator[dict]:
        stream = await litellm.acompletion(
            model=model,
            messages=messages,
            max_tokens=4096,
            temperature=0.7,
            stream=True,
        )

        async for chunk in stream:
            try:
                delta = chunk.choices[0].delta.content or ""
            except Exception:
                delta = ""
            if delta:
                yield {"type": "delta", "content": delta}

        yield {"type": "usage", "tokens": 0, "cost": 0.0}

    async def _ollama_stream(self, model: str, messages: list[dict]) -> AsyncIterator[dict]:
        if not self.config.ollama_api_key:
            raise ValueError("OLLAMA_API_KEY is required for ollama/* models")

        output_queue: queue.Queue = queue.Queue()
        done = object()

        def _produce():
            try:
                client = Client(
                    host=self.config.ollama_api_base,
                    headers={"Authorization": f"Bearer {self.config.ollama_api_key}"},
                )
                token_count = 0
                for chunk in client.chat(
                    model=model,
                    messages=messages,
                    stream=True,
                    options={"temperature": 0.7, "num_predict": 4096},
                ):
                    message = chunk.get("message", {}) if hasattr(chunk, "get") else getattr(chunk, "message", {})
                    if isinstance(message, dict):
                        delta = message.get("content", "")
                    else:
                        delta = getattr(message, "content", "")
                    if delta:
                        token_count += len(delta.split())
                        output_queue.put({"type": "delta", "content": delta})

                    is_done = chunk.get("done", False) if hasattr(chunk, "get") else getattr(chunk, "done", False)
                    if is_done:
                        prompt_tokens = int((chunk.get("prompt_eval_count") if hasattr(chunk, "get") else getattr(chunk, "prompt_eval_count", 0)) or 0)
                        output_tokens = int((chunk.get("eval_count") if hasattr(chunk, "get") else getattr(chunk, "eval_count", 0)) or 0)
                        output_queue.put({"type": "usage", "tokens": prompt_tokens + output_tokens or token_count})
                output_queue.put(done)
            except Exception as e:
                output_queue.put(e)
                output_queue.put(done)

        producer_task = asyncio.create_task(asyncio.to_thread(_produce))
        try:
            while True:
                item = await asyncio.to_thread(output_queue.get)
                if item is done:
                    break
                if isinstance(item, Exception):
                    raise item
                yield item
        finally:
            await producer_task

    def get_stats(self) -> dict:
        """Return brain usage statistics."""
        return {
            "total_tokens": self._total_tokens,
            "total_cost": round(self._total_cost, 4),
        }
