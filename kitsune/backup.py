"""
Kitsune Memory Backup — periodic export of ChromaDB collections to JSON and Markdown.

Runs in background, respects AUTO_BACKUP_INTERVAL_HOURS env var (default: 1 hour).
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger("kitsune.backup")


class MemoryBackup:
    """Periodic backup of interactions and knowledge collections."""

    def __init__(self, memory_system, data_dir: Path):
        self.memory = memory_system
        self.data_dir = Path(data_dir)
        self.backup_dir = self.data_dir / "backups"
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        self._running = False
        self._task = None

    async def start(self, interval_hours: float = 1.0):
        """Start the periodic backup background task."""
        if self._running:
            return
        self._running = True
        logger.info("💾 Memory backup started (every %.1fh)", interval_hours)
        import asyncio

        self._task = asyncio.create_task(self._loop(interval_hours))

    async def stop(self):
        """Stop the backup loop."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        logger.info("💾 Memory backup stopped.")

    async def _loop(self, interval_hours: float):
        import asyncio

        while self._running:
            try:
                self._backup_now()
            except Exception as e:
                logger.error("Memory backup failed: %s", e)
            await asyncio.sleep(interval_hours * 3600)

    def _backup_now(self):
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        json_path = self.backup_dir / f"memory_{timestamp}.json"
        md_path = self.backup_dir / f"memory_{timestamp}.md"

        backup_data: dict[str, Any] = {
            "exported_at": datetime.now(timezone.utc).isoformat(),
            "interactions": [],
            "knowledge": [],
        }

        # Export interactions
        try:
            int_results = self.memory.interactions.get()
            ids = int_results.get("ids", [])
            docs = int_results.get("documents", [])
            metas = int_results.get("metadatas", [])
            for i, doc in enumerate(docs):
                item = {
                    "id": ids[i] if i < len(ids) else "",
                    "document": doc,
                    "metadata": metas[i] if i < len(metas) and metas[i] else {},
                }
                backup_data["interactions"].append(item)
            logger.info("💾 Backed up %d interactions", len(docs))
        except Exception as e:
            logger.warning("Failed to backup interactions: %s", e)

        # Export knowledge
        try:
            know_results = self.memory.knowledge.get()
            ids = know_results.get("ids", [])
            docs = know_results.get("documents", [])
            metas = know_results.get("metadatas", [])
            for i, doc in enumerate(docs):
                item = {
                    "id": ids[i] if i < len(ids) else "",
                    "document": doc,
                    "metadata": metas[i] if i < len(metas) and metas[i] else {},
                }
                backup_data["knowledge"].append(item)
            logger.info("💾 Backed up %d knowledge items", len(docs))
        except Exception as e:
            logger.warning("Failed to backup knowledge: %s", e)

        # Write JSON
        try:
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(backup_data, f, ensure_ascii=False, indent=2)
            logger.info("💾 JSON backup saved: %s", json_path.name)
        except Exception as e:
            logger.error("Failed to write JSON backup: %s", e)

        # Write Markdown
        try:
            self._write_markdown(md_path, backup_data)
            logger.info("💾 Markdown backup saved: %s", md_path.name)
        except Exception as e:
            logger.error("Failed to write Markdown backup: %s", e)

        # Cleanup old backups (keep last 20)
        self._cleanup_old_backups(keep=20)

    def _write_markdown(self, path: Path, data: dict):
        lines = [
            "# Kitsune Memory Backup",
            f"",
            f"**Exported:** {data['exported_at']}",
            f"",
            f"---",
            f"",
            f"## Interactions ({len(data['interactions'])})",
            f"",
        ]
        for item in data["interactions"]:
            meta = item.get("metadata", {})
            user_id = meta.get("user_id", "unknown")
            cat = meta.get("task_category", "unknown")
            ts = meta.get("created_at", "")
            lines.append(f"### [{item.get('id', '?')}] User {user_id} | {cat} | {ts}")
            lines.append(f"{item.get('document', '')[:800]}")
            lines.append("")

        lines.extend([
            f"---",
            f"",
            f"## Knowledge ({len(data['knowledge'])})",
            f"",
        ])
        for item in data["knowledge"]:
            meta = item.get("metadata", {})
            user_id = meta.get("user_id", "unknown")
            ktype = meta.get("type", "fact")
            topic = meta.get("topic", "general")
            ts = meta.get("created_at", "")
            lines.append(f"### [{item.get('id', '?')}] User {user_id} | {ktype} | {topic} | {ts}")
            lines.append(f"{item.get('document', '')[:800]}")
            lines.append("")

        path.write_text("\n".join(lines), encoding="utf-8")

    def _cleanup_old_backups(self, keep: int = 20):
        try:
            backups = sorted(self.backup_dir.glob("memory_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
            for old in backups[keep:]:
                old.unlink(missing_ok=True)
                md = old.with_suffix(".md")
                md.unlink(missing_ok=True)
                logger.debug("Removed old backup: %s", old.name)
        except Exception as e:
            logger.warning("Backup cleanup failed: %s", e)

    def get_backups(self) -> list[Path]:
        """Return list of available JSON backup files."""
        return sorted(self.backup_dir.glob("memory_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
