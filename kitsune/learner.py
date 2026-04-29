"""
Kitsune Learner — Self-learning engine.
Extracts knowledge from interactions, optimizes routing, and evolves prompts.
"""

import logging
import re
from datetime import datetime, timezone
from pathlib import Path

from kitsune.config import Config
from kitsune.memory import MemorySystem
from kitsune.brain import Brain
from kitsune.router import Router

logger = logging.getLogger("kitsune.learner")


class Learner:
    """
    The self-learning engine that makes Kitsune smarter over time.

    It does three things:
    1. Extract and store knowledge from every interaction
    2. Process user feedback to improve model routing
    3. Periodically optimize routing rules based on performance data
    """

    def __init__(
        self,
        config: Config,
        memory: MemorySystem,
        brain: Brain,
        router: Router,
    ):
        self.config = config
        self.memory = memory
        self.brain = brain
        self.router = router
        self._interaction_counter = 0
        self._memory_markdown_path = Path(config.memory_markdown_path)
        self._ensure_memory_markdown()
        logger.info("📚 Learner initialized — ready to learn!")

    async def learn_from_interaction(
        self,
        user_id: int,
        user_message: str,
        bot_response: str,
        task_category: str,
        model_used: str,
        response_time: float,
        response_success: bool = True,
        interaction_id: str | None = None,
    ) -> str | None:
        """
        Process an interaction and extract learnings.
        Returns the interaction ID for feedback tracking.
        """
        self._interaction_counter += 1

        # 1. Store the raw interaction
        interaction_id = self.memory.store_interaction(
            user_id=user_id,
            user_message=user_message,
            bot_response=bot_response,
            task_category=task_category,
            model_used=model_used,
            response_time=response_time,
            interaction_id=interaction_id,
        )

        # 2. Record routing performance
        self.router.record_result(
            task_category=task_category,
            model_used=model_used,
            success=response_success,
            response_time=response_time,
        )
        self._append_memory_markdown(
            "interaction",
            [
                f"- user_id: `{user_id}`",
                f"- task: `{task_category}`",
                f"- model: `{model_used}`",
                f"- success: `{response_success}`",
                f"- response_time: `{response_time}`",
                "",
                "User:",
                self._fence(self._truncate(user_message, 900)),
                "",
                "Assistant:",
                self._fence(self._truncate(bot_response, 900)),
            ],
        )
        if not response_success:
            self._append_memory_markdown(
                "improvement-signal",
                [
                    f"- reason: weak/failed response heuristic",
                    f"- task: `{task_category}`",
                    f"- model: `{model_used}`",
                    "",
                    "User:",
                    self._fence(self._truncate(user_message, 1200)),
                ],
            )

        # 3. Extract memories automatically, with cheap direct rules first.
        direct_memories = self._extract_direct_memories(user_message)
        for memory in direct_memories:
            self.memory.store_knowledge(
                user_id=user_id,
                fact=memory["fact"],
                knowledge_type=memory["type"],
                topic=memory.get("topic", task_category),
                importance=memory.get("importance", 0.75),
                source="direct_rule",
            )
            self._append_learning_entry(
                user_id=user_id,
                source="direct_rule",
                memory_type=memory["type"],
                topic=memory.get("topic", task_category),
                fact=memory["fact"],
            )

        should_extract = (
            self.config.autonomous_learning
            and response_success
            and self._interaction_counter % self.config.auto_memory_extraction_every == 0
        )
        if should_extract:
            await self._extract_and_store_memories(
                user_id, user_message, bot_response
            )

        # 4. Periodic optimization
        if self._interaction_counter % 50 == 0:
            self._run_optimization()

        # 5. Update user preferences
        self._update_user_profile(user_id, task_category)

        return interaction_id

    async def learn_from_file(
        self,
        user_id: int,
        filename: str,
        file_kind: str,
        mime_type: str,
        text_preview: str,
        user_caption: str = "",
    ) -> str | None:
        """Store durable knowledge from a readable Telegram file upload."""
        if not self.config.auto_learn_from_files or not text_preview.strip():
            return None
        if self._looks_sensitive(text_preview):
            logger.info("Skipped auto-learning from sensitive-looking file: %s", filename)
            return None

        summary = self._summarize_file_preview(filename, file_kind, mime_type, text_preview)
        topic = "uploaded_file"
        if user_caption:
            topic = "uploaded_file_request"
            summary = f"{summary} User caption/request: {user_caption[:300]}"

        memory_id = self.memory.store_knowledge(
            user_id=user_id,
            fact=summary,
            knowledge_type="fact",
            topic=topic,
            importance=0.7,
            source="telegram_file",
        )
        self._append_learning_entry(
            user_id=user_id,
            source="telegram_file",
            memory_type="fact",
            topic=topic,
            fact=summary,
        )
        return memory_id

    async def process_feedback(
        self,
        user_id: int,
        interaction_id: str | None,
        feedback: str,
        task_category: str,
        model_used: str,
    ):
        """
        Process user feedback (positive/negative) on a response.
        This feeds back into routing optimization.
        """
        is_positive = feedback in ("positive", "good", "👍")

        # Update interaction record
        if interaction_id and not interaction_id.startswith("pending_"):
            self.memory.update_interaction_feedback(
                interaction_id, "positive" if is_positive else "negative"
            )

        # Update routing stats
        if not is_positive:
            self.router.record_result(
                task_category=task_category,
                model_used=model_used,
                success=False,
                response_time=0,
                from_feedback=True,
            )

            # Store correction as knowledge
            self.memory.store_knowledge(
                user_id=user_id,
                fact=f"User was not satisfied with {model_used} response for {task_category} task.",
                knowledge_type="correction",
                topic=task_category,
                importance=0.8,
                source="feedback",
            )

            logger.info(
                "👎 Negative feedback recorded for %s on %s",
                model_used,
                task_category,
            )
        else:
            logger.info(
                "👍 Positive feedback recorded for %s on %s",
                model_used,
                task_category,
            )

    async def _extract_and_store_memories(
        self,
        user_id: int,
        user_message: str,
        bot_response: str,
    ):
        """Extract notable facts from an interaction and store as knowledge."""
        try:
            memories = await self.brain.extract_memories(user_message, bot_response)

            for mem in memories:
                fact = mem.get("fact", "")
                mem_type = mem.get("type", "fact")

                if fact and len(fact) > 10:
                    self.memory.store_knowledge(
                        user_id=user_id,
                        fact=fact,
                        knowledge_type=mem_type,
                        topic=mem.get("topic", "general"),
                        importance=0.6,
                        source="llm_extraction",
                    )
                    self._append_learning_entry(
                        user_id=user_id,
                        source="llm_extraction",
                        memory_type=mem_type,
                        topic=mem.get("topic", "general"),
                        fact=fact,
                    )

            if memories:
                logger.info(
                    "🧠 Extracted %d memories from interaction", len(memories)
                )

        except Exception as e:
            logger.debug("Memory extraction skipped: %s", e)

    def _run_optimization(self):
        """Run periodic optimization of routing rules."""
        try:
            optimized = self.router.optimize_routing()
            if optimized:
                logger.info("🔄 Routing optimization complete!")
        except Exception as e:
            logger.error("Optimization failed: %s", e)

    def _update_user_profile(self, user_id: int, task_category: str):
        """Track user's task preferences."""
        prefs = self.config.learned_preferences
        uid = str(user_id)

        if uid not in prefs.get("users", {}):
            prefs.setdefault("users", {})[uid] = {
                "task_counts": {},
                "first_seen": datetime.now(timezone.utc).isoformat(),
                "total_interactions": 0,
            }

        user_prefs = prefs["users"][uid]
        user_prefs["total_interactions"] = user_prefs.get("total_interactions", 0) + 1
        task_counts = user_prefs.setdefault("task_counts", {})
        task_counts[task_category] = task_counts.get(task_category, 0) + 1
        user_prefs["last_seen"] = datetime.now(timezone.utc).isoformat()

        self.config.save_learned_preferences()

    @staticmethod
    def assess_response_success(response_text: str, response_success: bool = True) -> bool:
        """Cheap quality heuristic when explicit feedback buttons are disabled."""
        if not response_success:
            return False
        text = response_text.strip().lower()
        if len(text) < 8:
            return False
        weak_markers = (
            "all models failed",
            "masalah teknis",
            "coba lagi nanti",
            "i cannot access",
            "i can't access",
            "tidak bisa mengakses",
        )
        return not any(marker in text for marker in weak_markers)

    def save_user_profile(
        self,
        user_id: int,
        name: str = "",
        nickname: str = "",
        job: str = "",
        interests: str = "",
        preferences: str = "",
    ) -> list[str]:
        """Store a structured user profile as multiple knowledge entries."""
        stored: list[str] = []
        fields = [
            (name, "fact", "identity", "User's name is {value}.", 0.95),
            (nickname, "preference", "identity", "User prefers to be called {value}.", 0.95),
            (job, "fact", "profile", "User works as/is a {value}.", 0.85),
            (interests, "preference", "profile", "User is interested in {value}.", 0.8),
            (preferences, "preference", "profile", "User prefers {value}.", 0.8),
        ]
        for value, mem_type, topic, template, importance in fields:
            value = value.strip()
            if value:
                fact = template.format(value=value)
                mem_id = self.memory.store_knowledge(
                    user_id=user_id,
                    fact=fact,
                    knowledge_type=mem_type,
                    topic=topic,
                    importance=importance,
                    source="manual_teach",
                )
                if mem_id:
                    stored.append(mem_id)
                    self._append_learning_entry(
                        user_id=user_id,
                        source="manual_teach",
                        memory_type=mem_type,
                        topic=topic,
                        fact=fact,
                    )
        return stored

    def get_user_profile(self, user_id: int) -> dict:
        """Return structured user profile from knowledge store."""
        profile = {
            "name": None,
            "nickname": None,
            "job": None,
            "interests": None,
            "preferences": None,
        }
        try:
            results = self.memory.knowledge.get(
                where={
                    "$and": [
                        {"user_id": str(user_id)},
                        {"$or": [{"topic": "identity"}, {"topic": "profile"}]},
                    ]
                }
            )
        except Exception as e:
            logger.debug("Profile lookup failed: %s", e)
            return profile

        docs = results.get("documents", [])
        metas = results.get("metadatas", [])

        for doc, meta in zip(docs, metas):
            doc_lower = doc.lower()
            meta_topic = meta.get("topic") if meta else ""

            if meta_topic == "identity" or "name is" in doc_lower or "nama" in doc_lower:
                if not profile["name"]:
                    m = re.search(r"name is\s+([^,.!?]+)", doc, flags=re.IGNORECASE)
                    if m:
                        profile["name"] = m.group(1).strip()
                if not profile["nickname"]:
                    m = re.search(r"called\s+([^,.!?]+)", doc, flags=re.IGNORECASE)
                    if m:
                        profile["nickname"] = m.group(1).strip()
                    else:
                        m = re.search(r"panggil(?:\s+\w+)?\s+([^,.!?]+)", doc, flags=re.IGNORECASE)
                        if m:
                            profile["nickname"] = m.group(1).strip()

            if meta_topic == "profile" or "works as" in doc_lower or "is a" in doc_lower:
                if not profile["job"]:
                    m = re.search(r"(?:works as|is a)\s+([^,.!?]+)", doc, flags=re.IGNORECASE)
                    if m:
                        profile["job"] = m.group(1).strip()

            if meta_topic == "profile" and "interested in" in doc_lower:
                if not profile["interests"]:
                    m = re.search(r"interested in\s+([^,.!?]+)", doc, flags=re.IGNORECASE)
                    if m:
                        profile["interests"] = m.group(1).strip()

            if meta_topic == "profile" and "prefers" in doc_lower:
                if not profile["preferences"]:
                    m = re.search(r"prefers?\s+([^,.!?]+)", doc, flags=re.IGNORECASE)
                    if m:
                        profile["preferences"] = m.group(1).strip()

        return profile

    def teach_user(self, user_id: int, fact: str, topic: str = "manual") -> str | None:
        """Store an explicit user-taught memory."""
        memory_id = self.memory.store_knowledge(
            user_id=user_id,
            fact=fact,
            knowledge_type="preference",
            topic=topic,
            importance=0.95,
            source="manual_teach",
        )
        if memory_id:
            self._append_learning_entry(
                user_id=user_id,
                source="manual_teach",
                memory_type="preference",
                topic=topic,
                fact=fact,
            )
        return memory_id

    def _ensure_memory_markdown(self):
        if not self.config.auto_memory_markdown:
            return
        if self._memory_markdown_path.exists():
            return
        self._memory_markdown_path.parent.mkdir(parents=True, exist_ok=True)
        self._memory_markdown_path.write_text(
            "# Kitsune Runtime Memory Journal\n\n"
            "File ini dibuat otomatis sebagai bahan review dan improvement bot.\n"
            "Data sensitif umum seperti token/password/API key akan direduksi sebelum ditulis.\n\n",
            encoding="utf-8",
        )

    def _append_learning_entry(
        self,
        user_id: int,
        source: str,
        memory_type: str,
        topic: str,
        fact: str,
    ):
        self._append_memory_markdown(
            "learned-memory",
            [
                f"- user_id: `{user_id}`",
                f"- source: `{source}`",
                f"- type: `{memory_type}`",
                f"- topic: `{topic}`",
                "",
                self._fence(self._truncate(fact, 1200)),
            ],
        )

    def _append_memory_markdown(self, event_type: str, lines: list[str]):
        if not self.config.auto_memory_markdown:
            return
        try:
            self._ensure_memory_markdown()
            now = datetime.now(timezone.utc).isoformat()
            body = "\n".join(self._redact_sensitive(line) for line in lines)
            with self._memory_markdown_path.open("a", encoding="utf-8") as f:
                f.write(f"## {now} - {event_type}\n\n{body.strip()}\n\n")
        except Exception as e:
            logger.debug("Failed to append memory markdown: %s", e)

    @staticmethod
    def _redact_sensitive(text: str) -> str:
        redactions = [
            (r"(?i)(api[_-]?key|secret|token|password|passwd|private[_-]?key)\s*[:=]\s*['\"]?[^'\"\s]+", r"\1=<redacted>"),
            (r"-----BEGIN [^-]+PRIVATE KEY-----[\s\S]*?-----END [^-]+PRIVATE KEY-----", "<redacted-private-key>"),
            (r"\bsk-[A-Za-z0-9_\-]{20,}\b", "<redacted-api-key>"),
        ]
        redacted = text
        for pattern, replacement in redactions:
            redacted = re.sub(pattern, replacement, redacted)
        return redacted

    @staticmethod
    def _truncate(text: str, limit: int) -> str:
        cleaned = text.strip()
        if len(cleaned) <= limit:
            return cleaned
        return cleaned[:limit] + "\n...[truncated]"

    @staticmethod
    def _fence(text: str) -> str:
        return "```text\n" + text.replace("```", "` ` `") + "\n```"

    @staticmethod
    def _looks_sensitive(text: str) -> bool:
        patterns = (
            r"(?i)\b(api[_-]?key|secret|token|password|passwd|private[_-]?key)\b\s*[:=]",
            r"-----BEGIN (?:RSA |OPENSSH |EC |DSA )?PRIVATE KEY-----",
            r"\bsk-[A-Za-z0-9_\-]{20,}\b",
        )
        return any(re.search(pattern, text) for pattern in patterns)

    @staticmethod
    def _summarize_file_preview(filename: str, file_kind: str, mime_type: str, text_preview: str) -> str:
        clean_preview = " ".join(text_preview.strip().split())
        return (
            f"User uploaded readable file '{filename}' ({file_kind}, {mime_type}). "
            f"Important visible content preview: {clean_preview[:1200]}"
        )

    def get_user_profile_context(self, user_id: int) -> str:
        """Build stable user profile context for the system prompt."""
        prefs = self.config.learned_preferences.get("users", {}).get(str(user_id), {})
        recent_memories = self.memory.get_recent_knowledge(user_id, limit=8)
        identity = self.memory.get_user_identity(user_id)

        lines = ["[User profile learned by Kitsune]:"]

        name = identity.get("nickname") or identity.get("name")
        if name:
            lines.append(f"- user_name: {name}")

        if prefs:
            total = prefs.get("total_interactions", 0)
            lines.append(f"- total_interactions: {total}")
            task_counts = prefs.get("task_counts", {})
            if task_counts:
                common_tasks = sorted(task_counts.items(), key=lambda item: -item[1])[:3]
                summary = ", ".join(f"{task}={count}" for task, count in common_tasks)
                lines.append(f"- common_task_types: {summary}")

        for item in recent_memories:
            metadata = item.get("metadata", {})
            if metadata.get("type") in {"preference", "fact", "skill", "correction"}:
                lines.append(f"- {metadata.get('type')}: {item['document'][:220]}")

        return "\n".join(lines) if len(lines) > 1 else ""

    @staticmethod
    def _extract_direct_memories(user_message: str) -> list[dict]:
        """
        Cheap memory extraction for explicit user statements.
        This catches direct preferences immediately without waiting for an LLM call.
        """
        text = " ".join(user_message.strip().split())
        if len(text) < 4:
            return []

        patterns = [
            (
                r"\b(?:nama saya|namaku|nama aku|my name is)\s+([^,.!?\n]{2,80})",
                "fact",
                "identity",
                "User's name is {value}.",
                0.95,
            ),
            (
                r"\b(?:panggil saya|panggil aku|call me)\s+([^,.!?\n]{2,80})",
                "preference",
                "identity",
                "User prefers to be called {value}.",
                0.95,
            ),
            (
                r"\b(?:saya suka|aku suka|i like|i prefer)\s+([^,.!?\n]{2,160})",
                "preference",
                "preference",
                "User likes or prefers {value}.",
                0.8,
            ),
            (
                r"\b(?:saya tidak suka|aku tidak suka|i dislike|i don't like)\s+([^,.!?\n]{2,160})",
                "preference",
                "preference",
                "User dislikes {value}.",
                0.8,
            ),
            (
                r"\b(?:jangan|don't)\s+([^,.!?\n]{2,160})",
                "preference",
                "style",
                "User prefers that Kitsune does not {value}.",
                0.75,
            ),
            (
                r"\b(?:ingat bahwa|ingat kalau|remember that)\s+([^,.!?\n]{2,180})",
                "fact",
                "manual",
                "User explicitly asked Kitsune to remember: {value}.",
                0.9,
            ),
            (
                r"\b(?:gunakan bahasa|pakai bahasa|reply in|use language)\s+([^,.!?\n]{2,80})",
                "preference",
                "language",
                "User prefers language/style: {value}.",
                0.85,
            ),
            (
                r"\b(?:saya seorang|aku seorang|i am a[n]?|i work as|pekerjaan saya|kerjaan saya|kerjaan gue|kerjaan aku|gue kerja)\s+([^,.!?\n]{2,120})",
                "fact",
                "profile",
                "User works as/is a {value}.",
                0.85,
            ),
            (
                r"\b(?:minat saya|aku tertarik|i am interested in|saya tertarik|hobi saya|hobi gue|hobi aku)\s+([^,.!?\n]{2,160})",
                "preference",
                "profile",
                "User is interested in {value}.",
                0.8,
            ),
            (
                r"\b(?:preferensi saya|preferensi gue|aku prefer|i prefer)\s+([^,.!?\n]{2,160})",
                "preference",
                "profile",
                "User prefers {value}.",
                0.8,
            ),
        ]

        memories = []
        for pattern, mem_type, topic, template, importance in patterns:
            match = re.search(pattern, text, flags=re.IGNORECASE)
            if not match:
                continue
            value = match.group(1).strip(" .,!?:;\"'")
            if value:
                memories.append(
                    {
                        "fact": template.format(value=value),
                        "type": mem_type,
                        "topic": topic,
                        "importance": importance,
                    }
                )

        return memories

    def get_user_stats(self, user_id: int) -> str:
        """Get stats for a specific user."""
        prefs = self.config.learned_preferences
        uid = str(user_id)

        if uid not in prefs.get("users", {}):
            return "No data yet — keep chatting with me! 🦊"

        user = prefs["users"][uid]
        total = user.get("total_interactions", 0)
        task_counts = user.get("task_counts", {})
        first_seen = user.get("first_seen", "Unknown")

        # Find favorite task
        fav_task = max(task_counts, key=task_counts.get) if task_counts else "N/A"

        lines = [
            f"👤 **Your Stats**",
            f"Total chats: {total}",
            f"First seen: {first_seen[:10]}",
            f"Favorite topic: {fav_task}",
            "",
            "📋 **Task breakdown:**",
        ]

        for task, count in sorted(task_counts.items(), key=lambda x: -x[1]):
            pct = count / total * 100 if total > 0 else 0
            bar = "█" * int(pct / 10) + "░" * (10 - int(pct / 10))
            lines.append(f"  {task}: {bar} {count} ({pct:.0f}%)")

        return "\n".join(lines)
