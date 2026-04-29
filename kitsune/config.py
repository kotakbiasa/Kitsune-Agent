"""
Kitsune-Agent Configuration Manager.
Loads environment variables and manages dynamic config files.
"""

import os
import json
import logging
from pathlib import Path
from dotenv import load_dotenv

logger = logging.getLogger("kitsune.config")

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent
load_dotenv(PROJECT_ROOT / ".env")


def _env_path(name: str, default: Path) -> Path:
    raw_value = os.getenv(name, "").strip()
    return Path(raw_value) if raw_value else default


DATA_DIR = _env_path("KITSUNE_DATA_DIR", PROJECT_ROOT / "data")
MEMORY_DB_DIR = _env_path("KITSUNE_MEMORY_DB_DIR", PROJECT_ROOT / "memory_db")
SELF_IMPROVE_DIR = _env_path("KITSUNE_SELF_IMPROVE_DIR", DATA_DIR / "improvement_proposals")
TELEGRAM_UPLOAD_DIR = _env_path("KITSUNE_UPLOAD_DIR", DATA_DIR / "uploads")
MEMORY_MARKDOWN_PATH = _env_path("MEMORY_MARKDOWN_PATH", DATA_DIR / "memory.md")


def ensure_dirs():
    """Ensure all required directories exist."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    MEMORY_DB_DIR.mkdir(parents=True, exist_ok=True)
    SELF_IMPROVE_DIR.mkdir(parents=True, exist_ok=True)
    TELEGRAM_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    MEMORY_MARKDOWN_PATH.parent.mkdir(parents=True, exist_ok=True)


class Config:
    """Central configuration for Kitsune-Agent."""

    def __init__(self):
        ensure_dirs()
        self.data_dir = DATA_DIR

        # --- Telegram ---
        self.telegram_token: str = os.getenv("TELEGRAM_BOT_TOKEN", "")
        if not self.telegram_token:
            raise ValueError(
                "TELEGRAM_BOT_TOKEN is required. "
                "Get one from @BotFather on Telegram."
            )

        # --- Owner-only access ---
        raw_owner_ids = os.getenv("OWNER_USER_IDS", "").strip()
        self.owner_user_ids: list[int] = (
            [int(uid.strip()) for uid in raw_owner_ids.split(",") if uid.strip()]
            if raw_owner_ids
            else []
        )
        if not self.owner_user_ids:
            raise ValueError("OWNER_USER_IDS is required in owner-only mode.")
        raw_group_ids = os.getenv("APPROVED_GROUP_IDS", "").strip()
        self.approved_group_ids: list[int] = (
            [int(group_id.strip()) for group_id in raw_group_ids.split(",") if group_id.strip()]
            if raw_group_ids
            else []
        )
        self._runtime_groups = None

        # --- LLM API Keys ---
        self.openrouter_api_key = os.getenv("OPENROUTER_API_KEY", "")
        self.openai_api_key = os.getenv("OPENAI_API_KEY", "")
        self.anthropic_api_key = os.getenv("ANTHROPIC_API_KEY", "")
        self.gemini_api_key = os.getenv("GEMINI_API_KEY", "")
        self.ollama_api_key = os.getenv("OLLAMA_API_KEY", "")
        self.ollama_api_base = os.getenv("OLLAMA_API_BASE", "https://ollama.com").rstrip("/")
        self.ollama_fast_model = os.getenv("OLLAMA_FAST_MODEL", "ministral-3:3b-cloud")
        self.ollama_legacy_fast_model = os.getenv("OLLAMA_LEGACY_FAST_MODEL", "qwen3.5:cloud")
        self.ollama_reasoning_model = os.getenv("OLLAMA_REASONING_MODEL", "deepseek-v4-flash:cloud")
        self.ollama_coding_model = os.getenv("OLLAMA_CODING_MODEL", "qwen3-coder-next:cloud")
        self.ollama_productivity_model = os.getenv("OLLAMA_PRODUCTIVITY_MODEL", "minimax-m2.7:cloud")
        self.ollama_default_model = os.getenv("OLLAMA_DEFAULT_MODEL", self.ollama_fast_model)
        self.ollama_auto_discover_models = os.getenv("OLLAMA_AUTO_DISCOVER_MODELS", "true").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        ollama_all_cloud_models = [
            "ministral-3:3b-cloud",
            "rnj-1:8b-cloud",
            "ministral-3:8b-cloud",
            "ministral-3:14b-cloud",
            "qwen3-coder-next:cloud",
            "qwen3.5:cloud",
            "gemini-3-flash-preview:cloud",
            "nemotron-3-nano:30b-cloud",
            "devstral-small-2:24b-cloud",
            "gemma4:31b-cloud",
            "minimax-m2.7:cloud",
            "minimax-m2.5:cloud",
            "deepseek-v4-flash:cloud",
            "nemotron-3-super:cloud",
            "glm-5.1:cloud",
            "glm-5:cloud",
            "kimi-k2.6:cloud",
            "kimi-k2.5:cloud",
            "qwen3-next:80b-cloud",
            "devstral-2:123b-cloud",
            "deepseek-v4-pro:cloud",
            "qwen3.5:397b-cloud",
            "cogito-2.1:671b-cloud",
        ]
        self.ollama_model_pool = self._env_list(
            "OLLAMA_MODEL_POOL",
            ollama_all_cloud_models,
        )
        self.ollama_task_model_pools = {
            "simple_qa": self._env_list("OLLAMA_SIMPLE_MODELS", [self.ollama_fast_model, "rnj-1:8b-cloud", "ministral-3:8b-cloud", self.ollama_legacy_fast_model, "gemini-3-flash-preview:cloud"]),
            "complex_reasoning": self._env_list("OLLAMA_REASONING_MODELS", [self.ollama_reasoning_model, "nemotron-3-super:cloud", "glm-5.1:cloud", "gemini-3-flash-preview:cloud", "kimi-k2.6:cloud"]),
            "coding": self._env_list("OLLAMA_CODING_MODELS", [self.ollama_coding_model, "rnj-1:8b-cloud", "devstral-small-2:24b-cloud", "glm-5.1:cloud", "minimax-m2.7:cloud", "deepseek-v4-flash:cloud", "kimi-k2.6:cloud"]),
            "creative_writing": self._env_list("OLLAMA_CREATIVE_MODELS", [self.ollama_legacy_fast_model, "gemini-3-flash-preview:cloud", self.ollama_productivity_model, "kimi-k2.6:cloud"]),
            "math": self._env_list("OLLAMA_MATH_MODELS", [self.ollama_reasoning_model, "nemotron-3-super:cloud", "glm-5.1:cloud", "kimi-k2.6:cloud"]),
            "translation": self._env_list("OLLAMA_TRANSLATION_MODELS", [self.ollama_fast_model, self.ollama_legacy_fast_model, "gemini-3-flash-preview:cloud", "ministral-3:14b-cloud"]),
            "summarization": self._env_list("OLLAMA_SUMMARIZATION_MODELS", [self.ollama_legacy_fast_model, "gemini-3-flash-preview:cloud", self.ollama_productivity_model, "deepseek-v4-flash:cloud"]),
        }

        # Set keys in environment for LiteLLM
        if self.openrouter_api_key:
            os.environ["OPENROUTER_API_KEY"] = self.openrouter_api_key
        if self.openai_api_key:
            os.environ["OPENAI_API_KEY"] = self.openai_api_key
        if self.anthropic_api_key:
            os.environ["ANTHROPIC_API_KEY"] = self.anthropic_api_key
        if self.gemini_api_key:
            os.environ["GEMINI_API_KEY"] = self.gemini_api_key
        if self.ollama_api_key:
            os.environ["OLLAMA_API_KEY"] = self.ollama_api_key

        # --- Logging ---
        self.log_level = os.getenv("LOG_LEVEL", "INFO")

        # --- Telegram networking ---
        self.telegram_connect_timeout = self._env_float("TELEGRAM_CONNECT_TIMEOUT", 30.0)
        self.telegram_read_timeout = self._env_float("TELEGRAM_READ_TIMEOUT", 60.0)
        self.telegram_write_timeout = self._env_float("TELEGRAM_WRITE_TIMEOUT", 60.0)
        self.telegram_pool_timeout = self._env_float("TELEGRAM_POOL_TIMEOUT", 30.0)
        self.telegram_upload_dir = TELEGRAM_UPLOAD_DIR
        self.telegram_file_read_max_bytes = (
            max(1, min(int(self._env_float("TELEGRAM_FILE_READ_MAX_MB", 10.0)), 50))
            * 1024
            * 1024
        )

        # --- Response streaming ---
        self.enable_streaming = os.getenv("ENABLE_STREAMING", "true").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        self.telegram_stream_mode = os.getenv("TELEGRAM_STREAM_MODE", "draft").strip().lower()
        self.stream_edit_interval = max(1.0, self._env_float("STREAM_EDIT_INTERVAL", 1.0))
        self.stream_min_chars = max(12, int(self._env_float("STREAM_MIN_CHARS", 12.0)))
        self.telegram_chat_action_interval = max(
            1.0,
            self._env_float("TELEGRAM_CHAT_ACTION_INTERVAL", 4.0),
        )

        # --- Latency controls ---
        self.fast_routing = os.getenv("FAST_ROUTING", "true").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        self.background_learning = os.getenv("BACKGROUND_LEARNING", "true").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        self.autonomous_learning = os.getenv("AUTO_LEARNING", "true").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        self.auto_learn_from_files = os.getenv("AUTO_LEARN_FROM_FILES", "true").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        self.auto_memory_extraction_every = max(
            1,
            min(int(self._env_float("AUTO_MEMORY_EXTRACTION_EVERY", 1.0)), 20),
        )
        self.file_context_max_chars = max(
            500,
            min(int(self._env_float("FILE_CONTEXT_MAX_CHARS", 5000.0)), 20000),
        )
        self.auto_memory_markdown = os.getenv("AUTO_MEMORY_MARKDOWN", "true").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        self.memory_markdown_path = MEMORY_MARKDOWN_PATH

        # --- Safe self-improvement proposals ---
        self.enable_self_improve = os.getenv("ENABLE_SELF_IMPROVE", "true").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        self.self_improve_dir = SELF_IMPROVE_DIR

        # --- Dynamic config ---
        self._routing_rules = None
        self._prompt_templates = None
        self._learned_preferences = None

        logger.info("✅ Config loaded. Available providers: %s", self.available_providers)

    @property
    def available_providers(self) -> list[str]:
        """Return list of configured LLM providers."""
        providers = []
        if self.openrouter_api_key:
            providers.append("openrouter")
        if self.openai_api_key:
            providers.append("openai")
        if self.anthropic_api_key:
            providers.append("anthropic")
        if self.gemini_api_key:
            providers.append("gemini")
        if self.ollama_api_key:
            providers.append("ollama")
        return providers

    # ---- Dynamic Config Files ----

    def _load_json(self, filename: str, default: dict) -> dict:
        """Load a JSON config file, creating it with defaults if missing."""
        filepath = DATA_DIR / filename
        if filepath.exists():
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                logger.warning("Failed to load %s, using defaults: %s", filename, e)
        # Create with defaults
        self._save_json(filename, default)
        return default.copy()

    def _save_json(self, filename: str, data: dict):
        """Save data to a JSON config file."""
        filepath = DATA_DIR / filename
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    @property
    def routing_rules(self) -> dict:
        if self._routing_rules is None:
            self._routing_rules = self._load_json(
                "routing_rules.json", DEFAULT_ROUTING_RULES
            )
        return self._routing_rules

    def save_routing_rules(self):
        """Persist routing rules to disk."""
        if self._routing_rules:
            self._save_json("routing_rules.json", self._routing_rules)
            logger.info("💾 Routing rules saved (v%s)", self._routing_rules.get("version", "?"))

    @property
    def prompt_templates(self) -> dict:
        if self._prompt_templates is None:
            self._prompt_templates = self._load_json(
                "prompt_templates.json", DEFAULT_PROMPT_TEMPLATES
            )
        return self._prompt_templates

    def save_prompt_templates(self):
        if self._prompt_templates:
            self._save_json("prompt_templates.json", self._prompt_templates)

    @property
    def learned_preferences(self) -> dict:
        if self._learned_preferences is None:
            self._learned_preferences = self._load_json(
                "learned_preferences.json", {"users": {}}
            )
        return self._learned_preferences

    def save_learned_preferences(self):
        if self._learned_preferences:
            self._save_json("learned_preferences.json", self._learned_preferences)

    def is_owner(self, user_id: int) -> bool:
        return user_id in set(self.owner_user_ids)

    @property
    def runtime_groups(self) -> dict:
        if self._runtime_groups is None:
            self._runtime_groups = self._load_json("approved_groups.json", {"approved_group_ids": []})
        return self._runtime_groups

    def save_runtime_groups(self):
        if self._runtime_groups is not None:
            self._save_json("approved_groups.json", self._runtime_groups)

    @property
    def effective_approved_group_ids(self) -> set[int]:
        runtime_ids = self.runtime_groups.get("approved_group_ids", [])
        return {
            *self.approved_group_ids,
            *(int(group_id) for group_id in runtime_ids if str(group_id).strip()),
        }

    def is_group_approved(self, chat_id: int) -> bool:
        return chat_id in self.effective_approved_group_ids

    def add_approved_group(self, chat_id: int) -> bool:
        groups = self.runtime_groups
        ids = {int(group_id) for group_id in groups.get("approved_group_ids", [])}
        already_approved = chat_id in self.effective_approved_group_ids
        ids.add(chat_id)
        groups["approved_group_ids"] = sorted(ids)
        self.save_runtime_groups()
        return not already_approved

    def remove_approved_group(self, chat_id: int) -> bool:
        groups = self.runtime_groups
        ids = {int(group_id) for group_id in groups.get("approved_group_ids", [])}
        removed = chat_id in ids
        ids.discard(chat_id)
        groups["approved_group_ids"] = sorted(ids)
        self.save_runtime_groups()
        return removed

    @staticmethod
    def _env_float(name: str, default: float) -> float:
        raw_value = os.getenv(name, "").strip()
        if not raw_value:
            return default
        try:
            return float(raw_value)
        except ValueError:
            logger.warning("Invalid %s=%r, using %.1f", name, raw_value, default)
            return default

    @staticmethod
    def _env_list(name: str, default: list[str]) -> list[str]:
        raw_value = os.getenv(name, "").strip()
        values = raw_value.split(",") if raw_value else default
        cleaned = []
        seen = set()
        for value in values:
            item = value.strip()
            if not item or item in seen:
                continue
            cleaned.append(item)
            seen.add(item)
        return cleaned


# ===== Default Configs =====

DEFAULT_ROUTING_RULES = {
    "version": 1,
    "last_optimized": None,
    "total_interactions": 0,
    "optimization_threshold": 50,
    "rules": {
        "simple_qa": {
            "description": "Simple questions, greetings, factual lookups",
            "primary_model": "ollama/ministral-3:3b-cloud",
            "fallback_model": "ollama/rnj-1:8b-cloud",
            "success_count": 0,
            "fail_count": 0,
            "total_uses": 0,
            "avg_response_time": 0.0,
        },
        "complex_reasoning": {
            "description": "Analysis, comparison, multi-step reasoning",
            "primary_model": "ollama/deepseek-v4-flash:cloud",
            "fallback_model": "ollama/nemotron-3-super:cloud",
            "success_count": 0,
            "fail_count": 0,
            "total_uses": 0,
            "avg_response_time": 0.0,
        },
        "coding": {
            "description": "Code generation, debugging, code review",
            "primary_model": "ollama/qwen3-coder-next:cloud",
            "fallback_model": "ollama/rnj-1:8b-cloud",
            "success_count": 0,
            "fail_count": 0,
            "total_uses": 0,
            "avg_response_time": 0.0,
        },
        "creative_writing": {
            "description": "Stories, poems, creative content",
            "primary_model": "ollama/qwen3.5:cloud",
            "fallback_model": "ollama/gemini-3-flash-preview:cloud",
            "success_count": 0,
            "fail_count": 0,
            "total_uses": 0,
            "avg_response_time": 0.0,
        },
        "math": {
            "description": "Mathematical calculations, proofs, equations",
            "primary_model": "ollama/deepseek-v4-flash:cloud",
            "fallback_model": "ollama/nemotron-3-super:cloud",
            "success_count": 0,
            "fail_count": 0,
            "total_uses": 0,
            "avg_response_time": 0.0,
        },
        "translation": {
            "description": "Language translation tasks",
            "primary_model": "ollama/ministral-3:3b-cloud",
            "fallback_model": "ollama/qwen3.5:cloud",
            "success_count": 0,
            "fail_count": 0,
            "total_uses": 0,
            "avg_response_time": 0.0,
        },
        "summarization": {
            "description": "Summarizing text, documents, articles",
            "primary_model": "ollama/qwen3.5:cloud",
            "fallback_model": "ollama/gemini-3-flash-preview:cloud",
            "success_count": 0,
            "fail_count": 0,
            "total_uses": 0,
            "avg_response_time": 0.0,
        },
    },
}

DEFAULT_PROMPT_TEMPLATES = {
    "version": 1,
    "system_prompt": (
        "Kamu adalah Kitsune 🦊, asisten AI yang cerdas, adaptif, dan punya memori jangka panjang. "
        "Kamu selalu menjawab dalam bahasa yang sama dengan yang digunakan user. "
        "Gunakan memori pengguna hanya jika relevan, jangan mengarang memori yang tidak ada, "
        "dan jangan menyebut detail memori secara eksplisit kecuali memang membantu jawaban. "
        "Kamu ramah, helpful, dan memberikan jawaban yang detail namun tidak bertele-tele. "
        "Jika kamu tidak tahu sesuatu, akui dengan jujur."
    ),
    "classifier_prompt": (
        "Classify the following user message into exactly ONE category. "
        "Categories: simple_qa, complex_reasoning, coding, creative_writing, math, translation, summarization. "
        "Reply with ONLY the category name, nothing else.\n\n"
        "User message: {message}"
    ),
    "memory_extraction_prompt": (
        "Extract only durable facts, preferences, corrections, or skills worth remembering "
        "for future conversations with this user. Do not store one-off requests. "
        "Return valid JSON only: a list of objects with fact, type, and topic fields. "
        "type must be one of preference, fact, correction, skill. "
        "If nothing durable is present, return [].\n\n"
        "User: {user_message}\nAssistant: {assistant_response}"
    ),
}
