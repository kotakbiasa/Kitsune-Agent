"""
Kitsune Router — Task classifier and intelligent model switcher.
Classifies incoming messages and routes them to the optimal AI model.
"""

import logging
import re
import time
from datetime import datetime, timezone

import litellm
from ollama import Client

from kitsune.config import Config
from kitsune.model_utils import pick_fast_model, pick_model_for_task

logger = logging.getLogger("kitsune.router")

# Valid task categories
TASK_CATEGORIES = [
    "simple_qa",
    "complex_reasoning",
    "coding",
    "creative_writing",
    "math",
    "translation",
    "summarization",
    "web_search",
    "multimodal",
    "long_context",
]


class Router:
    """Classifies tasks and selects the best model."""

    # Circuit breaker: consecutive failures before tripping
    CB_FAILURE_THRESHOLD = 3
    CB_COOLDOWN_SECONDS = 600  # 10 minutes
    # Auto-discovery TTL
    DISCOVERY_TTL_SECONDS = 3600  # 1 hour

    def __init__(self, config: Config):
        self.config = config
        self._discovered_ollama_models: list[str] | None = None
        self._last_discovery_at: float = 0.0
        self._circuit_breakers: dict[str, dict] = {}
        self._classifier_model = pick_fast_model(config)
        logger.info("🧭 Router initialized. Classifier model: %s", self._classifier_model)

    async def classify_task(self, message: str) -> tuple[str, float]:
        """
        Classify a user message into a task category.
        Returns (category, confidence).
        """
        if self.config.fast_routing:
            category = self._classify_task_locally(message)
            logger.info("📋 Locally classified as: %s", category)
            return category, 0.75

        classifier_prompt = self.config.prompt_templates.get(
            "classifier_prompt",
            (
                "Classify the following user message into exactly ONE category. "
                "Categories: simple_qa, complex_reasoning, coding, creative_writing, math, translation, summarization. "
                "Reply with ONLY the category name, nothing else.\n\n"
                "User message: {message}"
            ),
        )

        try:
            response = await litellm.acompletion(
                model=self._classifier_model,
                messages=[
                    {
                        "role": "user",
                        "content": classifier_prompt.format(message=message[:500]),
                    }
                ],
                max_tokens=20,
                temperature=0.0,
            )

            category = response.choices[0].message.content.strip().lower()

            # Validate category
            if category in TASK_CATEGORIES:
                logger.info("📋 Classified as: %s (model: %s)", category, self._classifier_model)
                return category, 0.9
            else:
                # Try to fuzzy match
                for cat in TASK_CATEGORIES:
                    if cat in category:
                        logger.info("📋 Fuzzy matched as: %s", cat)
                        return cat, 0.7

                logger.warning(
                    "⚠️ Unknown category '%s', defaulting to simple_qa", category
                )
                return "simple_qa", 0.3

        except Exception as e:
            logger.error("Classification failed: %s. Defaulting to simple_qa.", e)
            return "simple_qa", 0.2

    @staticmethod
    def _classify_task_locally(message: str) -> str:
        text = message.lower()
        stripped = text.strip()

        # Web search detection
        search_keywords = (
            r"\b(cari|cariin|search|google|lookup|find)\b.*\b(info|informasi|tentang|about)\b",
            r"\b(berita|news|update|terbaru|latest)\b",
            r"\b(cuaca|weather|harga bitcoin|btc|harga emas|gold price)\b",
            r"\b(jadwal|schedule|skor|score|pertandingan)\b",
            r"\b(berapa|how much|what is)\b.*\b(harga|price|nilai|value)\b",
            r"\b(kapan|when)\b.*\b(rilis|release|launch|rilis|rilis)\b",
        )
        for pattern in search_keywords:
            if re.search(pattern, text):
                return "web_search"

        if re.search(r"```|def |class |function |const |let |var |npm |pip |docker|traceback|error|bug|kode|coding|script|api|json|yaml|regex", text):
            return "coding"
        if re.search(r"\b(translate|terjemah|terjemahkan|alih bahasa)\b", text):
            return "translation"
        if re.search(r"\b(summary|summarize|ringkas|rangkum|resume)\b", text):
            return "summarization"
        if re.search(r"\b(hitung|calculate|math|matematika|persamaan|equation|integral|turunan)\b|[0-9]\s*[\+\-\*/^]\s*[0-9]", text):
            return "math"
        if re.search(r"\b(cerita|puisi|novel|copywriting|caption|creative|story|poem)\b", text):
            return "creative_writing"
        # Multimodal detection
        multimodal_keywords = (
            r"\b(gambar|image|photo|foto|video|visual|lihat|describe.*image|analyze.*photo)\b",
            r"\b(ocr|read.*image|scan.*photo|apa.*ini.*gambar)\b",
        )
        for pattern in multimodal_keywords:
            if re.search(pattern, text):
                return "multimodal"

        # Long context detection
        long_context_keywords = (
            r"\b(dokumen.*panjang|long.*document|banyak.*teks|analyze.*pdf|baca.*file.*besar)\b",
            r"\b(100k|200k|context.*long|very.*long|ringkas.*buku|summarize.*book)\b",
        )
        for pattern in long_context_keywords:
            if re.search(pattern, text):
                return "long_context"

        if len(stripped) > 180 or re.search(r"\b(analisa|analyze|compare|bandingkan|kenapa|mengapa|strategi|arsitektur|rancang)\b", text):
            return "complex_reasoning"
        return "simple_qa"

    def get_model_for_task(self, task_category: str) -> tuple[str, list[str]]:
        """
        Get the best model for a task category.
        Uses USE_CASE_MODELS pool with auto-switch based on available providers.
        Returns (primary_model, fallback_models).
        """
        # Use use-case based model picking (cloud-first)
        try:
            primary, fallbacks = pick_model_for_task(self.config, task_category)
            logger.info(
                "⚡ Model for '%s': %s (fallbacks: %s)",
                task_category,
                primary,
                ", ".join(fallbacks) if fallbacks else "-",
            )
            return primary, fallbacks
        except ValueError:
            # No cloud providers configured; fall through to Ollama legacy path
            logger.warning("No cloud LLM providers configured, trying Ollama fallback")

        # Legacy Ollama path (kept for backward compatibility)
        if "ollama" in self.config.available_providers:
            candidates = self._ollama_candidates_for_task(task_category)
            if candidates:
                logger.info("⚡ Ollama candidates for '%s': %s", task_category, ", ".join(candidates))
                return candidates[0], candidates[1:]

        logger.error("No LLM providers configured!")
        return "none", []

    def _ollama_candidates_for_task(self, task_category: str) -> list[str]:
        configured = self.config.ollama_task_model_pools.get(task_category) or self.config.ollama_model_pool
        configured = [self._as_ollama_model(model) for model in configured]

        discovered = [self._as_ollama_model(model) for model in self._discover_ollama_models()]
        global_pool = [self._as_ollama_model(model) for model in self.config.ollama_model_pool]

        stats = self.config.routing_rules.get("rules", {}).get(task_category, {}).get("model_stats", {})
        candidates = [*configured, *global_pool, *discovered]
        candidates = [model for model in self._dedupe(candidates) if self._validate_model(model).startswith("ollama/")]

        # Sort by dynamic score (circuit breaker aware)
        candidates.sort(key=lambda model: self._model_score(stats.get(model, {}), model) or -999, reverse=True)

        # Log circuit breaker status
        for model in candidates:
            cb = self._circuit_breakers.get(model, {})
            if cb.get("tripped_until", 0) > time.time():
                logger.info("⛔ Model %s is circuit-breaker skipped", model)

        if not candidates:
            candidates = [self._as_ollama_model(self.config.ollama_default_model)]
        return candidates

    def _discover_ollama_models(self) -> list[str]:
        if not self.config.ollama_auto_discover_models or not self.config.ollama_api_key:
            return []
        now = time.time()
        if self._discovered_ollama_models is not None and (now - self._last_discovery_at) < self.DISCOVERY_TTL_SECONDS:
            return self._discovered_ollama_models

        try:
            client = Client(
                host=self.config.ollama_api_base,
                headers={"Authorization": f"Bearer {self.config.ollama_api_key}"},
            )
            response = client.list()
            raw_models = response.get("models", []) if hasattr(response, "get") else getattr(response, "models", [])
            names = []
            for item in raw_models or []:
                if isinstance(item, dict):
                    name = item.get("name") or item.get("model")
                else:
                    name = getattr(item, "name", None) or getattr(item, "model", None)
                if name:
                    names.append(str(name))
            self._discovered_ollama_models = self._dedupe(names)
            self._last_discovery_at = now
            logger.info("Discovered %d Ollama model(s)", len(self._discovered_ollama_models))
        except Exception as e:
            logger.info("Ollama model discovery skipped: %s", str(e)[:160])
            self._discovered_ollama_models = []
            self._last_discovery_at = now
        return self._discovered_ollama_models

    @staticmethod
    def _as_ollama_model(model: str) -> str:
        model = model.strip()
        if not model:
            return ""
        if model.startswith("ollama/"):
            return model
        return f"ollama/{model}"

    @staticmethod
    def _dedupe(models: list[str]) -> list[str]:
        output = []
        seen = set()
        for model in models:
            if not model or model in seen:
                continue
            output.append(model)
            seen.add(model)
        return output

    def _validate_model(self, model: str) -> str:
        """Check if we have credentials for a model's provider."""
        provider_map = {
            "ollama/": "ollama",
            "openrouter/": "openrouter",
            "openai/": "openai",
            "anthropic/": "anthropic",
            "gemini/": "gemini",
        }

        for prefix, provider in provider_map.items():
            if model.startswith(prefix):
                if provider in self.config.available_providers:
                    return model
                else:
                    # Provider not available, try to find alternative
                    logger.warning(
                        "Provider '%s' not configured, finding alternative for %s",
                        provider,
                        model,
                    )
                    return pick_fast_model(self.config)

        return model

    def record_result(
        self,
        task_category: str,
        model_used: str,
        success: bool,
        response_time: float,
        from_feedback: bool = False,
        error_type: str | None = None,
    ):
        """Record the result of a model execution for optimization.
        error_type: rate_limit, timeout, context_length, auth_failed, content_policy, unknown
        """
        rules = self.config.routing_rules
        if task_category not in rules.get("rules", {}):
            return

        # Update circuit breaker
        cb = self._circuit_breakers.setdefault(model_used, {"consecutive_fails": 0, "tripped_until": 0.0})
        if success:
            cb["consecutive_fails"] = 0
        else:
            cb["consecutive_fails"] = cb.get("consecutive_fails", 0) + 1
            if cb["consecutive_fails"] >= self.CB_FAILURE_THRESHOLD:
                cb["tripped_until"] = time.time() + self.CB_COOLDOWN_SECONDS
                logger.warning(
                    "🔴 Circuit breaker tripped for %s (%d consecutive fails). Cooling down for %ds",
                    model_used,
                    cb["consecutive_fails"],
                    self.CB_COOLDOWN_SECONDS,
                )

        rule = rules["rules"][task_category]
        rule["total_uses"] = rule.get("total_uses", 0) + 1

        if success:
            rule["success_count"] = rule.get("success_count", 0) + 1
        else:
            rule["fail_count"] = rule.get("fail_count", 0) + 1

        # Update moving average response time
        if not from_feedback and response_time > 0:
            prev_avg = rule.get("avg_response_time", 0.0)
            timed_uses = rule.get("timed_uses", 0) + 1
            rule["avg_response_time"] = round(
                (prev_avg * (timed_uses - 1) + response_time) / timed_uses, 3
            )
            rule["timed_uses"] = timed_uses

        model_stats = rule.setdefault("model_stats", {})
        model = model_stats.setdefault(
            model_used,
            {
                "uses": 0,
                "success_count": 0,
                "fail_count": 0,
                "avg_response_time": 0.0,
                "timed_uses": 0,
                "error_counts": {},
                "last_error_type": None,
            },
        )
        model["uses"] = model.get("uses", 0) + 1
        model["last_used"] = time.time()
        if success:
            model["success_count"] = model.get("success_count", 0) + 1
        else:
            model["fail_count"] = model.get("fail_count", 0) + 1
            if error_type:
                model["last_error_type"] = error_type
                err_counts = model.setdefault("error_counts", {})
                err_counts[error_type] = err_counts.get(error_type, 0) + 1

                # Error pattern learning: if same error repeats 3x, auto-penalize
                if err_counts[error_type] >= 3:
                    logger.warning(
                        "📉 Error pattern detected for %s: %s occurred %d times",
                        model_used,
                        error_type,
                        err_counts[error_type],
                    )
        if not from_feedback and response_time > 0:
            timed_uses = model.get("timed_uses", 0) + 1
            prev_avg = model.get("avg_response_time", 0.0)
            model["avg_response_time"] = round(
                (prev_avg * (timed_uses - 1) + response_time) / timed_uses, 3
            )
            model["timed_uses"] = timed_uses

        # Update global counter
        rules["total_interactions"] = rules.get("total_interactions", 0) + 1

        # Save periodically
        if rules["total_interactions"] % 10 == 0:
            self.config.save_routing_rules()

        logger.debug(
            "📊 Recorded: %s on %s (success=%s, time=%.2fs, error=%s)",
            task_category,
            model_used,
            success,
            response_time,
            error_type or "none",
        )

    def optimize_routing(self):
        """
        Analyze performance data and optimize model routing.
        Called periodically by the learner.
        """
        rules = self.config.routing_rules
        threshold = rules.get("optimization_threshold", 50)

        if rules.get("total_interactions", 0) < threshold:
            return False  # Not enough data

        logger.info("🔄 Optimizing routing rules (interactions: %d)...", rules["total_interactions"])

        # Reset expired circuit breakers during optimization
        now = time.time()
        reset_models = [
            model for model, cb in self._circuit_breakers.items()
            if cb.get("tripped_until", 0) <= now
        ]
        for model in reset_models:
            logger.info("🟢 Circuit breaker reset for %s", model)
            self._circuit_breakers[model]["consecutive_fails"] = 0
            self._circuit_breakers[model]["tripped_until"] = 0.0

        # For each category, check if current model is performing well.
        for category, rule in rules.get("rules", {}).items():
            total = rule.get("total_uses", 0)
            if total < 5:
                continue

            success_rate = rule.get("success_count", 0) / total if total > 0 else 0
            model_stats = rule.get("model_stats", {})
            primary = rule.get("primary_model")
            fallback = rule.get("fallback_model")

            primary_score = self._model_score(model_stats.get(primary, {}), primary)
            fallback_score = self._model_score(model_stats.get(fallback, {}), fallback)
            route_changed = False

            if fallback_score is not None and (
                primary_score is None or fallback_score > primary_score + 0.15
            ):
                old_primary = rule["primary_model"]
                rule["primary_model"] = rule["fallback_model"]
                rule["fallback_model"] = old_primary
                route_changed = True
                logger.info(
                    "🔀 Promoted fallback for '%s': %s → %s",
                    category,
                    old_primary,
                    rule["primary_model"],
                )

            # If success rate is below 60%, force a primary/fallback swap as a last resort.
            elif success_rate < 0.6:
                old_primary = rule["primary_model"]
                rule["primary_model"] = rule["fallback_model"]
                rule["fallback_model"] = old_primary
                route_changed = True
                logger.info(
                    "🔀 Swapped weak route for '%s': %s → %s",
                    category,
                    old_primary,
                    rule["primary_model"],
                )

            if route_changed:
                # Reset counters
                rule["success_count"] = 0
                rule["fail_count"] = 0
                rule["total_uses"] = 0
                rule["avg_response_time"] = 0.0
                rule["timed_uses"] = 0

        # Update version and timestamp
        rules["version"] = rules.get("version", 1) + 1
        rules["last_optimized"] = datetime.now(timezone.utc).isoformat()
        rules["total_interactions"] = 0  # Reset counter

        self.config.save_routing_rules()
        logger.info("✅ Routing rules optimized to v%d", rules["version"])
        return True

    def _model_score(self, stats: dict, model: str) -> float | None:
        """Score model quality from feedback, latency, and error patterns. Higher is better."""
        # Circuit breaker penalty
        cb = self._circuit_breakers.get(model, {})
        tripped_until = cb.get("tripped_until", 0.0)
        if tripped_until > time.time():
            return -999.0  # Force skip

        uses = stats.get("uses", 0)
        if uses < 2:
            # Give new models a small positive bias for exploration
            return 0.05

        success_rate = stats.get("success_count", 0) / uses
        avg_time = stats.get("avg_response_time", 0.0)
        latency_penalty = min(avg_time / 60, 0.25) if avg_time else 0.0

        # Recency penalty: if model hasn't been used in last 24h, slight penalty
        last_used = stats.get("last_used", 0)
        recency_penalty = 0.0
        if last_used and (time.time() - last_used) > 86400:
            recency_penalty = 0.05

        # Error pattern penalty
        error_penalty = 0.0
        err_counts = stats.get("error_counts", {})
        for err_type, count in err_counts.items():
            if count >= 3:
                # Escalating penalty: 0.05 per error type that has hit threshold
                error_penalty += 0.05 * min(count / 3, 3)

        return success_rate - latency_penalty - recency_penalty - error_penalty

    def get_routing_stats(self) -> str:
        """Get human-readable routing statistics."""
        rules = self.config.routing_rules
        lines = [
            f"📊 **Kitsune Routing Stats** (v{rules.get('version', 1)})",
            f"Total interactions: {rules.get('total_interactions', 0)}",
            f"Last optimized: {rules.get('last_optimized', 'Never')}",
            "",
        ]

        for category, rule in rules.get("rules", {}).items():
            total = rule.get("total_uses", 0)
            success = rule.get("success_count", 0)
            rate = f"{(success / total * 100):.0f}%" if total > 0 else "N/A"
            avg_time = rule.get("avg_response_time", 0)

            emoji = "🟢" if total == 0 or (success / total > 0.7 if total > 0 else True) else "🟡"
            lines.append(
                f"{emoji} **{category}**: {rule.get('primary_model', '?').split('/')[-1]} "
                f"(uses: {total}, success: {rate}, avg: {avg_time:.1f}s)"
            )

        return "\n".join(lines)
