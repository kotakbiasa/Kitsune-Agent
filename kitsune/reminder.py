"""
Kitsune Reminder System — asyncio-based scheduler for user reminders.
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Callable, Awaitable

logger = logging.getLogger("kitsune.reminder")


@dataclass
class Reminder:
    id: str
    user_id: int
    chat_id: int
    text: str
    due_at: datetime
    created_at: datetime
    sent: bool = False


class ReminderSystem:
    """Persistent reminder scheduler with JSON backing."""

    def __init__(
        self,
        storage_path: Path,
        callback: Callable[[Reminder], Awaitable[None]],
        tick_seconds: float = 30.0,
    ):
        self.storage_path = storage_path
        self._callback = callback
        self._tick_seconds = tick_seconds
        self._reminders: list[Reminder] = []
        self._task: asyncio.Task | None = None
        self._load()

    def start(self):
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self._loop())

    def stop(self):
        if self._task and not self._task.done():
            self._task.cancel()

    def _load(self):
        if not self.storage_path.exists():
            return
        try:
            raw = json.loads(self.storage_path.read_text(encoding="utf-8"))
            for item in raw.get("reminders", []):
                self._reminders.append(
                    Reminder(
                        id=item["id"],
                        user_id=item["user_id"],
                        chat_id=item["chat_id"],
                        text=item["text"],
                        due_at=datetime.fromisoformat(item["due_at"]),
                        created_at=datetime.fromisoformat(item["created_at"]),
                        sent=item.get("sent", False),
                    )
                )
        except Exception as e:
            logger.warning("Failed to load reminders: %s", e)

    def _save(self):
        try:
            data = {
                "reminders": [
                    {
                        "id": r.id,
                        "user_id": r.user_id,
                        "chat_id": r.chat_id,
                        "text": r.text,
                        "due_at": r.due_at.isoformat(),
                        "created_at": r.created_at.isoformat(),
                        "sent": r.sent,
                    }
                    for r in self._reminders
                ]
            }
            self.storage_path.write_text(
                json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
            )
        except Exception as e:
            logger.warning("Failed to save reminders: %s", e)

    async def _loop(self):
        while True:
            try:
                await asyncio.sleep(self._tick_seconds)
            except asyncio.CancelledError:
                break
            now = datetime.now(timezone.utc)
            triggered: list[Reminder] = []
            for reminder in self._reminders:
                if not reminder.sent and reminder.due_at <= now:
                    triggered.append(reminder)
            for reminder in triggered:
                reminder.sent = True
                try:
                    await self._callback(reminder)
                except Exception as e:
                    logger.error("Reminder callback failed: %s", e)
            if triggered:
                self._save()

    def add(self, user_id: int, chat_id: int, text: str, due_at: datetime) -> Reminder:
        reminder = Reminder(
            id=datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S") + f"_{user_id}",
            user_id=user_id,
            chat_id=chat_id,
            text=text,
            due_at=due_at,
            created_at=datetime.now(timezone.utc),
        )
        self._reminders.append(reminder)
        self._save()
        logger.info("Reminder added for user %s at %s", user_id, due_at.isoformat())
        return reminder

    def list_for_user(self, user_id: int) -> list[Reminder]:
        return [r for r in self._reminders if r.user_id == user_id and not r.sent]

    def cancel(self, user_id: int, reminder_id: str) -> bool:
        for i, r in enumerate(self._reminders):
            if r.id == reminder_id and r.user_id == user_id and not r.sent:
                self._reminders.pop(i)
                self._save()
                return True
        return False

    @staticmethod
    def parse_time(text: str) -> datetime | None:
        """Parse relative time like '10m', '1h30m', '2d', or absolute ISO."""
        text = text.strip().lower()
        text = (
            text.replace("menit", "m")
            .replace("jam", "h")
            .replace("detik", "s")
            .replace("hari", "d")
        )

        m = re.match(r"^(\d+d)?\s*(\d+h)?\s*(\d+m)?\s*(\d+s)?$", text)
        if m:
            days = int(m.group(1)[:-1]) if m.group(1) else 0
            hours = int(m.group(2)[:-1]) if m.group(2) else 0
            minutes = int(m.group(3)[:-1]) if m.group(3) else 0
            seconds = int(m.group(4)[:-1]) if m.group(4) else 0
            if any((days, hours, minutes, seconds)):
                return datetime.now(timezone.utc) + timedelta(
                    days=days, hours=hours, minutes=minutes, seconds=seconds
                )

        try:
            return datetime.fromisoformat(text)
        except ValueError:
            pass

        return None
