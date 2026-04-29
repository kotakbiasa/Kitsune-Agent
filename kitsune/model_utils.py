"""
Kitsune Model Utilities — shared LLM provider selection logic.
"""
from __future__ import annotations

from kitsune.config import Config


def pick_fast_model(config: Config) -> str:
    """
    Pick the cheapest/fastest available model for classification and extraction tasks.
    Raises ValueError if no provider is configured.
    """
    candidates = [
        ("ollama", f"ollama/{config.ollama_fast_model}"),
        ("openrouter", "openrouter/google/gemini-2.0-flash-001"),
        ("openrouter", "openrouter/openai/gpt-4o-mini"),
        ("gemini", "gemini/gemini-2.0-flash"),
        ("openai", "openai/gpt-4o-mini"),
    ]
    for provider, model in candidates:
        if provider in config.available_providers:
            return model

    if config.available_providers:
        provider = config.available_providers[0]
        fallbacks = {
            "openrouter": "openrouter/google/gemini-2.0-flash-001",
            "openai": "openai/gpt-4o-mini",
            "anthropic": "anthropic/claude-sonnet-4-20250514",
            "gemini": "gemini/gemini-2.0-flash",
            "ollama": f"ollama/{config.ollama_fast_model}",
        }
        return fallbacks.get(provider, "openrouter/google/gemini-2.0-flash-001")

    raise ValueError(
        "No cloud LLM provider configured. Set OLLAMA_API_KEY, OPENROUTER_API_KEY, "
        "OPENAI_API_KEY, GEMINI_API_KEY, or ANTHROPIC_API_KEY in .env"
    )
