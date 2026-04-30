"""
Kitsune Health Monitor — Periodic provider health checks.

Pings configured LLM providers every 5 minutes, tracks uptime,
and auto-adjusts available provider pool.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

import litellm
from ollama import Client

from kitsune.config import Config
from kitsune.model_utils import USE_CASE_MODELS, pick_fast_model

logger = logging.getLogger("kitsune.health")

HEALTH_CHECK_INTERVAL = 300  # 5 minutes
HEALTH_LOG_MAX_ENTRIES = 500


class HealthMonitor:
    """Periodically checks LLM provider health and updates runtime state."""

    def __init__(self, config: Config, router=None):
        self.config = config
        self.router = router
        self._task: asyncio.Task | None = None
        self._stop_event = asyncio.Event()
        self._log_path = config.data_dir / "health_log.json"
        self._last_results: dict[str, dict] = {}

    def start(self):
        """Start the background health check loop."""
        if self._task is None or self._task.done():
            self._stop_event.clear()
            self._task = asyncio.create_task(self._loop())
            logger.info("🏥 Health monitor started (interval: %ds)", HEALTH_CHECK_INTERVAL)

    def stop(self):
        """Stop the background health check loop."""
        self._stop_event.set()
        if self._task and not self._task.done():
            self._task.cancel()
            logger.info("🏥 Health monitor stopped")

    async def _loop(self):
        """Main loop: run checks periodically."""
        # Initial delay to let bot finish startup
        await asyncio.sleep(30)
        while not self._stop_event.is_set():
            try:
                await self.run_check()
            except Exception as e:
                logger.warning("Health check round failed: %s", e)
            try:
                await asyncio.wait_for(self._stop_event.wait(), timeout=HEALTH_CHECK_INTERVAL)
            except asyncio.TimeoutError:
                pass

    async def run_check(self):
        """Run a single health check round for all providers."""
        providers = self.config.available_providers.copy()
        if not providers:
            logger.warning("No providers configured, skipping health check")
            return

        results = {}
        for provider in providers:
            healthy, latency, error = await self._check_provider(provider)
            results[provider] = {
                "healthy": healthy,
                "latency_ms": round(latency * 1000, 1),
                "error": error,
                "checked_at": datetime.now(timezone.utc).isoformat(),
            }
            status = "✅" if healthy else "❌"
            logger.info(
                "%s Health check %s: %.0fms%s",
                status,
                provider,
                latency * 1000,
                f" ({error})" if error else "",
            )

        self._last_results = results
        self._append_log(results)

        # Auto-reset circuit breakers for healthy providers
        if self.router:
            for provider, result in results.items():
                if result["healthy"]:
                    reset_any = False
                    for model, cb in list(self.router._circuit_breakers.items()):
                        if model.startswith(f"{provider}/"):
                            if cb.get("tripped_until", 0) > time.time():
                                cb["tripped_until"] = 0.0
                                cb["consecutive_fails"] = 0
                                reset_any = True
                                logger.info("🟢 Circuit breaker auto-reset for %s (health check pass)", model)
                    if reset_any:
                        self.config.save_routing_rules()

    async def _check_provider(self, provider: str) -> tuple[bool, float, str | None]:
        """Check a single provider. Returns (healthy, latency_seconds, error_or_none)."""
        # Pick a fast/cheap model for this provider
        pool = USE_CASE_MODELS.get("fast_response", USE_CASE_MODELS.get("simple_qa", {}))
        model = pool.get(provider)
        if not model:
            # Fallback: try to construct a minimal model string
            model = f"{provider}/test"

        start = time.time()
        try:
            if provider == "ollama":
                if not self.config.ollama_api_key:
                    return False, 0.0, "No OLLAMA_API_KEY"

                def _ping():
                    client = Client(
                        host=self.config.ollama_api_base,
                        headers={"Authorization": f"Bearer {self.config.ollama_api_key}"},
                    )
                    # Just list models as a lightweight operation
                    client.list()

                await asyncio.wait_for(asyncio.to_thread(_ping), timeout=15)
                return True, time.time() - start, None
            else:
                if provider == "custom_codex":
                    if not self.config.custom_codex_base_url or not self.config.custom_codex_api_key:
                        return False, 0.0, "No CUSTOM_CODEX_BASE_URL or CUSTOM_CODEX_API_KEY"
                    response = await asyncio.wait_for(
                        litellm.acompletion(
                            model=f"openai/{self.config.custom_codex_model}",
                            messages=[{"role": "user", "content": "ping"}],
                            max_tokens=1,
                            temperature=0.0,
                            api_base=self.config.custom_codex_base_url,
                            api_key=self.config.custom_codex_api_key,
                        ),
                        timeout=15,
                    )
                else:
                    # Cloud provider: send a minimal completion
                    response = await asyncio.wait_for(
                        litellm.acompletion(
                            model=model,
                            messages=[{"role": "user", "content": "ping"}],
                            max_tokens=1,
                            temperature=0.0,
                        ),
                        timeout=15,
                    )
                # Verify we got a non-empty response
                if response.choices and response.choices[0].message.content is not None:
                    return True, time.time() - start, None
                return False, time.time() - start, "Empty response"

        except asyncio.TimeoutError:
            return False, time.time() - start, "Timeout"
        except Exception as e:
            return False, time.time() - start, str(e)[:120]

    def _append_log(self, results: dict):
        """Append health results to JSON log with rotation."""
        try:
            entries = []
            if self._log_path.exists():
                try:
                    with open(self._log_path, "r", encoding="utf-8") as f:
                        entries = json.load(f)
                except (json.JSONDecodeError, IOError):
                    entries = []

            entries.append({
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "results": results,
            })

            # Rotate old entries
            if len(entries) > HEALTH_LOG_MAX_ENTRIES:
                entries = entries[-HEALTH_LOG_MAX_ENTRIES:]

            with open(self._log_path, "w", encoding="utf-8") as f:
                json.dump(entries, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.debug("Failed to write health log: %s", e)

    def get_summary(self) -> str:
        """Human-readable health summary."""
        if not self._last_results:
            return "🏥 Health check belum dijalankan."

        lines = ["🏥 **Provider Health**"]
        for provider, result in self._last_results.items():
            emoji = "🟢" if result["healthy"] else "🔴"
            lines.append(
                f"{emoji} **{provider}**: {result['latency_ms']}ms"
                + (f" ({result['error']})" if result['error'] else "")
            )
        return "\n".join(lines)
