"""
Kitsune Model Utilities — shared LLM provider selection logic.

Cloud-first model picking by use case.
"""
from __future__ import annotations

from kitsune.config import Config

# ---- Use-case based model pools (cloud only) ----

USE_CASE_MODELS = {
    "coding": {
        "ollama": "ollama/deepseek-3.2",
        "openai": "openai/gpt-5.3-codex",
        "openrouter": "openrouter/qwen/qwen3-coder-next",
        "anthropic": "anthropic/claude-sonnet-4-20250514",
        "gemini": "gemini/gemini-2.5-flash-002",
    },
    "complex_reasoning": {
        "anthropic": "anthropic/claude-opus-4-6-20251001",
        "openai": "openai/gpt-5.4",
        "ollama": "ollama/deepseek-v3-2-volc",
        "openrouter": "openrouter/deepseek/deepseek-v3-2-volc",
        "gemini": "gemini/gemini-2.5-pro-002",
    },
    "creative_writing": {
        "anthropic": "anthropic/claude-sonnet-4-5-20251001",
        "openai": "openai/gpt-5.2",
        "ollama": "ollama/qwen3.5",
        "gemini": "gemini/gemini-2.5-flash-002",
        "openrouter": "openrouter/anthropic/claude-sonnet-4.5",
    },
    "fast_response": {
        "anthropic": "anthropic/claude-haiku-4-5-20251001",
        "gemini": "gemini/gemini-3.1-flash-lite-002",
        "openai": "openai/gpt-4o-mini",
        "ollama": "ollama/ministral-3:3b-cloud",
        "openrouter": "openrouter/google/gemini-3.1-flash-lite",
    },
    "multimodal": {
        "gemini": "gemini/gemini-2.5-pro-002",
        "openai": "openai/gpt-5.4",
        "ollama": "ollama/gemma4:31b-cloud",
        "anthropic": "anthropic/claude-opus-4-6-20251001",
        "openrouter": "openrouter/google/gemini-2.5-pro",
    },
    "long_context": {
        "openrouter": "openrouter/kimi/kimi-k2.5",
        "gemini": "gemini/gemini-2.5-pro-002",
        "anthropic": "anthropic/claude-opus-4-6-20251001",
        "openai": "openai/gpt-5.4",
        "ollama": "ollama/gemma4:31b-cloud",
    },
    "math": {
        "ollama": "ollama/deepseek-v3-2-volc",
        "openai": "openai/gpt-5.4",
        "anthropic": "anthropic/claude-opus-4-6-20251001",
        "gemini": "gemini/gemini-2.5-pro-002",
        "openrouter": "openrouter/deepseek/deepseek-v3-2-volc",
    },
    "translation": {
        "gemini": "gemini/gemini-3.1-flash-lite-002",
        "openai": "openai/gpt-4o-mini",
        "ollama": "ollama/ministral-3:3b-cloud",
        "anthropic": "anthropic/claude-haiku-4-5-20251001",
        "openrouter": "openrouter/google/gemini-3.1-flash-lite",
    },
    "summarization": {
        "ollama": "ollama/qwen3.5",
        "openai": "openai/gpt-5.2",
        "gemini": "gemini/gemini-2.5-flash-002",
        "anthropic": "anthropic/claude-sonnet-4-5-20251001",
        "openrouter": "openrouter/anthropic/claude-sonnet-4.5",
    },
    "simple_qa": {
        "ollama": "ollama/ministral-3:3b-cloud",
        "gemini": "gemini/gemini-3.1-flash-lite-002",
        "openai": "openai/gpt-4o-mini",
        "anthropic": "anthropic/claude-haiku-4-5-20251001",
        "openrouter": "openrouter/google/gemini-3.1-flash-lite",
    },
    "web_search": {
        "anthropic": "anthropic/claude-opus-4-6-20251001",
        "openai": "openai/gpt-5.4",
        "ollama": "ollama/deepseek-v3-2-volc",
        "gemini": "gemini/gemini-2.5-pro-002",
        "openrouter": "openrouter/deepseek/deepseek-v3-2-volc",
    },
}


def pick_model_for_task(config: Config, task_category: str) -> tuple[str, list[str]]:
    """
    Pick the best model for a task category based on available providers.
    Returns (primary_model, fallback_models).
    """
    pool = USE_CASE_MODELS.get(task_category, USE_CASE_MODELS["simple_qa"])
    return _select_from_pool(config, pool)


def pick_fast_model(config: Config) -> str:
    """
    Pick the cheapest/fastest available model for classification and extraction tasks.
    """
    pool = USE_CASE_MODELS["fast_response"]
    primary, fallbacks = _select_from_pool(config, pool)
    return primary


def _select_from_pool(config: Config, pool: dict[str, str]) -> tuple[str, list[str]]:
    """Select primary + fallbacks from a model pool based on available providers."""
    candidates = []
    for provider, model in pool.items():
        if provider in config.available_providers:
            candidates.append(model)

    if not candidates:
        # Fallback to any available provider
        if config.available_providers:
            first_provider = config.available_providers[0]
            default_model = pool.get(first_provider, "openrouter/google/gemini-3.1-flash-lite")
            return default_model, []
        raise ValueError(
            "No cloud LLM provider configured. Set OLLAMA_API_KEY, OPENROUTER_API_KEY, "
            "OPENAI_API_KEY, GEMINI_API_KEY, or ANTHROPIC_API_KEY in .env"
        )

    return candidates[0], candidates[1:]


def get_model_info(model: str) -> dict:
    """Return human-readable info about a model."""
    info_map = {
        "deepseek-3.2": {"name": "DeepSeek 3.2", "use": "Coding", "provider": "Ollama"},
        "gpt-5.3-codex": {"name": "GPT-5.3 Codex", "use": "Coding", "provider": "OpenAI"},
        "qwen3-coder-next": {"name": "Qwen3 Coder", "use": "Coding", "provider": "Ollama"},
        "claude-opus-4.6": {"name": "Claude Opus 4.6", "use": "Complex Reasoning", "provider": "Anthropic"},
        "gpt-5.4": {"name": "GPT-5.4", "use": "Flagship / Multimodal", "provider": "OpenAI"},
        "deepseek-v3-2-volc": {"name": "DeepSeek v3-2 Volc", "use": "Fast Reasoning", "provider": "Ollama"},
        "claude-sonnet-4.5": {"name": "Claude Sonnet 4.5", "use": "Writing", "provider": "Anthropic"},
        "claude-sonnet-4": {"name": "Claude Sonnet 4", "use": "Writing", "provider": "Anthropic"},
        "gpt-5.2": {"name": "GPT-5.2", "use": "Writing / General", "provider": "OpenAI"},
        "claude-haiku-4.5": {"name": "Claude Haiku 4.5", "use": "Fast Response", "provider": "Anthropic"},
        "gemini-3.1-flash-lite": {"name": "Gemini 3.1 Flash Lite", "use": "Fast Response", "provider": "Gemini"},
        "gemini-2.5-flash": {"name": "Gemini 2.5 Flash", "use": "Fast Response", "provider": "Gemini"},
        "gemini-2.5-pro": {"name": "Gemini 2.5 Pro", "use": "Multimodal / Long Context", "provider": "Gemini"},
        "gemini-3.1-pro": {"name": "Gemini 3.1 Pro", "use": "Multimodal", "provider": "Gemini"},
        "kimi-k2.5": {"name": "Kimi k2.5", "use": "Long Context (200K)", "provider": "OpenRouter"},
    }

    for key, value in info_map.items():
        if key in model:
            return value

    return {"name": model.split("/")[-1], "use": "General", "provider": "Unknown"}
