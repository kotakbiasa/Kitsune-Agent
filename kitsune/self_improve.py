"""
Safe self-improvement proposal system.

This module records improvement ideas without modifying production code.
"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path

from kitsune.brain import Brain
from kitsune.config import Config

logger = logging.getLogger("kitsune.self_improve")


class SelfImprover:
    """Create reviewable improvement proposals from commands and feedback."""

    def __init__(self, config: Config, brain: Brain):
        self.config = config
        self.brain = brain
        self.proposal_dir = config.self_improve_dir
        self.proposal_dir.mkdir(parents=True, exist_ok=True)

    async def propose_from_request(self, user_id: int, request: str) -> Path:
        """Generate a reviewable implementation proposal for a user request."""
        prompt = (
            "You are preparing a safe code improvement proposal for Kitsune-Agent.\n"
            "Do not claim that code has been changed. Do not include secrets.\n"
            "Return a concise Markdown proposal with these sections:\n"
            "1. Goal\n"
            "2. Suspected files to inspect\n"
            "3. Proposed patch approach\n"
            "4. Validation commands\n"
            "5. Risk notes\n\n"
            f"User request:\n{request}"
        )
        model = f"ollama/{self.config.ollama_coding_model}"
        fallback = [
            f"ollama/{name}" if not name.startswith("ollama/") else name
            for name in self.config.ollama_task_model_pools.get("coding", self.config.ollama_model_pool)
            if name != self.config.ollama_coding_model
        ]
        response = await self.brain.think(
            user_message=prompt,
            model=model,
            fallback_model=fallback,
            memory_context="",
            conversation_history=[],
        )
        body = response.content if response.success else self._fallback_body(request, response.error)
        return self._write_proposal(
            title=request[:90],
            source="manual",
            body=body,
            metadata={
                "user_id": user_id,
                "model_used": response.model_used,
                "response_time": response.response_time,
            },
        )

    def record_negative_feedback(
        self,
        user_id: int,
        user_message: str,
        bot_response: str,
        task_category: str,
        model_used: str,
    ) -> Path:
        """Record negative feedback as an improvement candidate."""
        body = (
            "## Goal\n"
            "Investigate and improve an answer that received negative feedback.\n\n"
            "## Context\n"
            f"- Task category: `{task_category}`\n"
            f"- Model used: `{model_used}`\n"
            f"- User id: `{user_id}`\n\n"
            "## User Message\n"
            f"{self._fence(user_message)}\n\n"
            "## Bot Response\n"
            f"{self._fence(bot_response[:4000])}\n\n"
            "## Proposed Patch Approach\n"
            "- Review whether routing selected the right model.\n"
            "- Check whether prompt, memory context, or streaming finalization caused answer quality issues.\n"
            "- Add or adjust a focused test/manual validation case before applying changes.\n\n"
            "## Validation Commands\n"
            "```bash\n"
            "uv run python -m compileall main.py kitsune\n"
            "uv lock --check\n"
            "```\n"
        )
        return self._write_proposal(
            title=f"Negative feedback: {task_category}",
            source="negative_feedback",
            body=body,
            metadata={
                "user_id": user_id,
                "task_category": task_category,
                "model_used": model_used,
            },
        )

    def list_recent(self, limit: int = 8) -> list[Path]:
        """Return recent proposal files, newest first."""
        files = sorted(
            self.proposal_dir.glob("*.md"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        return files[:limit]

    def _write_proposal(self, title: str, source: str, body: str, metadata: dict) -> Path:
        now = datetime.now(timezone.utc)
        slug = self._slug(title)
        path = self.proposal_dir / f"{now.strftime('%Y%m%d-%H%M%S')}-{slug}.md"
        header = {
            "created_at": now.isoformat(),
            "source": source,
            **metadata,
        }
        content = (
            f"# {title.strip() or 'Improvement Proposal'}\n\n"
            "```json\n"
            f"{json.dumps(header, indent=2, ensure_ascii=False)}\n"
            "```\n\n"
            f"{body.strip()}\n"
        )
        path.write_text(content, encoding="utf-8")
        logger.info("Self-improvement proposal saved: %s", path)
        return path

    @staticmethod
    def _slug(text: str) -> str:
        slug = re.sub(r"[^a-zA-Z0-9]+", "-", text.lower()).strip("-")
        return slug[:60] or "proposal"

    @staticmethod
    def _fence(text: str) -> str:
        return "```text\n" + text.replace("```", "` ` `") + "\n```"

    @staticmethod
    def _fallback_body(request: str, error: str | None) -> str:
        return (
            "## Goal\n"
            f"{request}\n\n"
            "## Proposed Patch Approach\n"
            "- Inspect the related code path manually.\n"
            "- Make the smallest focused patch.\n"
            "- Validate with compile and runtime smoke tests.\n\n"
            "## Risk Notes\n"
            f"LLM proposal generation failed: {error or 'unknown error'}\n"
        )
