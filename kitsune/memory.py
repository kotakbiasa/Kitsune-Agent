"""
Kitsune Memory System — ChromaDB-based long-term memory.
Stores interactions, learns preferences, and provides context for future responses.
"""

import logging
import math
import re
import uuid
from datetime import datetime, timezone
from hashlib import sha1

import chromadb

from kitsune.config import MEMORY_DB_DIR

logger = logging.getLogger("kitsune.memory")


class HashEmbeddingFunction:
    """
    Deterministic cloud-safe embeddings.

    This avoids downloading/running local ML embedding models inside Docker/VPS.
    It is less semantically rich than a hosted embedding API, but it is stable,
    cheap, and good enough for lightweight memory recall.
    """

    def __init__(self, dimensions: int = 384):
        self.dimensions = dimensions

    def name(self) -> str:
        return "kitsune_hash_embedding"

    def __call__(self, input: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in input]

    def embed_query(
        self,
        query: str | list[str] | None = None,
        input: str | list[str] | None = None,
    ) -> list[float] | list[list[float]]:
        value = input if input is not None else query
        if isinstance(value, list):
            return [self._embed(item) for item in value]
        return self._embed(value or "")

    def embed_documents(
        self,
        documents: list[str] | None = None,
        input: list[str] | None = None,
    ) -> list[list[float]]:
        texts = input if input is not None else documents or []
        return [self._embed(document) for document in texts]

    def _embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        tokens = re.findall(r"[\w']+", text.lower())
        for token in tokens:
            digest = sha1(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "big") % self.dimensions
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[index] += sign

        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0:
            return vector
        return [round(value / norm, 6) for value in vector]


class MemorySystem:
    """Persistent memory using ChromaDB for semantic search."""

    def __init__(self):
        logger.info("🧠 Initializing memory system at %s", MEMORY_DB_DIR)

        self.client = chromadb.PersistentClient(path=str(MEMORY_DB_DIR))

        self.embedding_fn = HashEmbeddingFunction()

        # Main collections
        self.interactions = self.client.get_or_create_collection(
            name="interactions",
            embedding_function=self.embedding_fn,
            metadata={"description": "All user-bot interactions"},
        )

        self.knowledge = self.client.get_or_create_collection(
            name="knowledge",
            embedding_function=self.embedding_fn,
            metadata={"description": "Learned facts, preferences, and skills"},
        )

        logger.info(
            "✅ Memory loaded: %d interactions, %d knowledge items",
            self.interactions.count(),
            self.knowledge.count(),
        )

    # ---- Store ----

    def store_interaction(
        self,
        user_id: int,
        user_message: str,
        bot_response: str,
        task_category: str,
        model_used: str,
        response_time: float,
        interaction_id: str | None = None,
    ):
        """Store a complete interaction for future reference."""
        doc = f"User asked: {user_message}\nBot answered: {bot_response[:500]}"
        mem_id = interaction_id or f"int_{uuid.uuid4().hex[:12]}"

        try:
            self.interactions.add(
                documents=[doc],
                ids=[mem_id],
                metadatas=[
                    {
                        "user_id": str(user_id),
                        "task_category": task_category,
                        "model_used": model_used,
                        "response_time": response_time,
                        "created_at": datetime.now(timezone.utc).isoformat(),
                        "user_message_preview": user_message[:200],
                        "feedback": "none",  # Will be updated if user gives feedback
                    }
                ],
            )
            logger.debug("💾 Stored interaction %s for user %s", mem_id, user_id)
            return mem_id
        except Exception as e:
            logger.error("Failed to store interaction: %s", e)
            return None

    def store_knowledge(
        self,
        user_id: int,
        fact: str,
        knowledge_type: str = "fact",
        topic: str = "general",
        importance: float = 0.5,
        source: str = "extracted",
    ):
        """Store a learned fact, preference, or skill."""
        fact = " ".join(fact.strip().split())
        if not fact:
            return None

        now = datetime.now(timezone.utc).isoformat()
        fact_hash = sha1(f"{user_id}:{fact.lower()}".encode("utf-8")).hexdigest()[:16]

        existing = self._find_knowledge_by_hash(user_id, fact_hash)
        if existing:
            metadata = existing.get("metadata", {}).copy()
            metadata.update(
                {
                    "importance": max(float(metadata.get("importance", 0.0)), importance),
                    "updated_at": now,
                    "last_seen": now,
                    "source": source,
                    "access_count": int(metadata.get("access_count", 0)),
                }
            )
            try:
                self.knowledge.update(
                    ids=[existing["id"]],
                    documents=[fact],
                    metadatas=[metadata],
                )
                logger.debug("🧠 Refreshed knowledge: %s", fact[:80])
                return existing["id"]
            except Exception as e:
                logger.error("Failed to refresh knowledge: %s", e)
                return None

        mem_id = f"know_{uuid.uuid4().hex[:12]}"

        try:
            self.knowledge.add(
                documents=[fact],
                ids=[mem_id],
                metadatas=[
                    {
                        "user_id": str(user_id),
                        "type": knowledge_type,
                        "topic": topic,
                        "importance": importance,
                        "source": source,
                        "fact_hash": fact_hash,
                        "created_at": now,
                        "updated_at": now,
                        "last_seen": now,
                        "access_count": 0,
                    }
                ],
            )
            logger.debug("🧠 Stored knowledge: %s (type=%s)", fact[:80], knowledge_type)
            return mem_id
        except Exception as e:
            logger.error("Failed to store knowledge: %s", e)
            return None

    # ---- Retrieve ----

    def recall_relevant(
        self,
        query: str,
        user_id: int | None = None,
        n_results: int = 5,
        collection: str = "both",
    ) -> list[dict]:
        """
        Retrieve memories relevant to the query.
        Returns list of dicts with 'document', 'metadata', 'distance'.
        """
        results = []

        where_filter = {"user_id": str(user_id)} if user_id else None

        try:
            if collection in ("both", "interactions"):
                int_results = self.interactions.query(
                    query_texts=[query],
                    n_results=min(n_results, max(self.interactions.count(), 1)),
                    where=where_filter if self.interactions.count() > 0 and where_filter else None,
                )
                results.extend(self._format_results(int_results, "interaction"))

            if collection in ("both", "knowledge"):
                know_results = self.knowledge.query(
                    query_texts=[query],
                    n_results=min(n_results, max(self.knowledge.count(), 1)),
                    where=where_filter if self.knowledge.count() > 0 and where_filter else None,
                )
                results.extend(self._format_results(know_results, "knowledge"))

        except Exception as e:
            logger.error("Memory recall failed: %s", e)

        # Sort by relevance (lower distance = more relevant)
        results.sort(key=lambda x: x.get("distance", 999))
        return results[:n_results]

    def get_user_context(self, user_id: int, current_message: str) -> str:
        """
        Build a context string from relevant memories for a user.
        This gets injected into the LLM prompt.
        """
        memories = self.recall_relevant(
            query=current_message, user_id=user_id, n_results=5
        )

        if not memories:
            return ""

        context_parts = ["[Relevant memories about this user]:"]
        for mem in memories:
            if mem["distance"] < 1.5:  # Only include reasonably relevant memories
                if mem["source"] == "knowledge":
                    meta = mem.get("metadata", {})
                    prefix = f"- {meta.get('type', 'memory')}"
                    context_parts.append(f"{prefix}: {mem['document'][:300]}")
                    self._increment_knowledge_access(mem)
                else:
                    context_parts.append(f"- previous_interaction: {mem['document'][:300]}")

        if len(context_parts) == 1:
            return ""  # No relevant memories found

        return "\n".join(context_parts)

    # ---- Feedback ----

    def update_interaction_feedback(self, interaction_id: str, feedback: str):
        """Update feedback for a stored interaction (positive/negative)."""
        try:
            existing = self.interactions.get(ids=[interaction_id])
            metadata = (
                existing.get("metadatas", [{}])[0].copy()
                if existing.get("metadatas")
                else {}
            )
            metadata["feedback"] = feedback
            metadata["feedback_at"] = datetime.now(timezone.utc).isoformat()
            self.interactions.update(
                ids=[interaction_id],
                metadatas=[metadata],
            )
            logger.info("📊 Updated feedback for %s: %s", interaction_id, feedback)
        except Exception as e:
            logger.error("Failed to update feedback: %s", e)

    def get_recent_knowledge(self, user_id: int, limit: int = 10) -> list[dict]:
        """Return recent learned knowledge for a user."""
        try:
            results = self.knowledge.get(where={"user_id": str(user_id)})
        except Exception as e:
            logger.error("Failed to read recent knowledge: %s", e)
            return []

        items = []
        ids = results.get("ids", [])
        docs = results.get("documents", [])
        metas = results.get("metadatas", [])
        for i, doc in enumerate(docs):
            metadata = metas[i] if i < len(metas) and metas[i] else {}
            items.append(
                {
                    "id": ids[i] if i < len(ids) else "",
                    "document": doc,
                    "metadata": metadata,
                }
            )

        items.sort(
            key=lambda item: item["metadata"].get("updated_at")
            or item["metadata"].get("created_at")
            or "",
            reverse=True,
        )
        return items[:limit]

    # ---- Stats ----

    def get_user_identity(self, user_id: int) -> dict:
        """Return identity info (name, nickname) for a user from knowledge store."""
        try:
            results = self.knowledge.get(
                where={
                    "$and": [
                        {"user_id": str(user_id)},
                        {"type": {"$in": ["fact", "preference"]}},
                    ]
                }
            )
        except Exception as e:
            logger.debug("Identity lookup failed: %s", e)
            return {}

        name = None
        nickname = None
        docs = results.get("documents", [])
        metas = results.get("metadatas", [])

        for doc, meta in zip(docs, metas):
            doc_lower = doc.lower()
            if "name is" in doc_lower or "nama" in doc_lower:
                if not name:
                    m = re.search(r"(?:name is|nama(?:ku| saya| aku)?\s+)([^,.!?]+)", doc, flags=re.IGNORECASE)
                    if m:
                        name = m.group(1).strip()
            if "prefer" in doc_lower or "panggil" in doc_lower:
                if not nickname:
                    m = re.search(r"(?:called|panggil)(?:\s+\w+)?\s+([^,.!?]+)", doc, flags=re.IGNORECASE)
                    if m:
                        nickname = m.group(1).strip()

        return {"name": name, "nickname": nickname}

    # ---- Personality ----

    def set_user_personality(self, user_id: int, personality: str) -> str | None:
        """Store a user's preferred bot personality as knowledge."""
        personality = personality.strip()
        if not personality:
            return None
        fact = f"Bot personality for this user: {personality}"
        return self.store_knowledge(
            user_id=user_id,
            fact=fact,
            knowledge_type="preference",
            topic="personality",
            importance=0.95,
            source="user_command",
        )

    def get_user_personality(self, user_id: int) -> str:
        """Return the user's preferred personality text, or empty string."""
        try:
            results = self.knowledge.get(
                where={
                    "$and": [
                        {"user_id": str(user_id)},
                        {"topic": "personality"},
                    ]
                }
            )
        except Exception as e:
            logger.debug("Personality lookup failed: %s", e)
            return ""

        docs = results.get("documents", [])
        metas = results.get("metadatas", [])
        if not docs:
            return ""

        # Get most recently updated
        latest = ""
        latest_time = ""
        for doc, meta in zip(docs, metas):
            updated = meta.get("updated_at") or meta.get("created_at") or ""
            if updated >= latest_time:
                latest_time = updated
                latest = doc

        # Strip the boilerplate prefix
        prefix = "Bot personality for this user: "
        if latest.startswith(prefix):
            return latest[len(prefix):]
        return latest

    def get_stats(self) -> dict:
        """Return memory system statistics."""
        return {
            "total_interactions": self.interactions.count(),
            "total_knowledge": self.knowledge.count(),
        }

    # ---- Cleanup ----

    def forget_user(self, user_id: int) -> int:
        """Delete all memories for a specific user. Returns count deleted."""
        deleted = 0
        for collection in [self.interactions, self.knowledge]:
            try:
                # Get all items for this user
                results = collection.get(
                    where={"user_id": str(user_id)},
                )
                if results["ids"]:
                    collection.delete(ids=results["ids"])
                    deleted += len(results["ids"])
            except Exception as e:
                logger.error("Error deleting memories for user %s: %s", user_id, e)
        logger.info("🗑️ Deleted %d memories for user %s", deleted, user_id)
        return deleted

    # ---- Helpers ----

    @staticmethod
    def _format_results(raw_results: dict, source: str) -> list[dict]:
        """Format ChromaDB query results into a clean list."""
        formatted = []
        if not raw_results or not raw_results.get("documents"):
            return formatted

        for i, doc in enumerate(raw_results["documents"][0]):
            formatted.append(
                {
                    "document": doc,
                    "metadata": (
                        raw_results["metadatas"][0][i]
                        if raw_results.get("metadatas")
                        else {}
                    ),
                    "distance": (
                        raw_results["distances"][0][i]
                        if raw_results.get("distances")
                        else 999
                    ),
                    "id": raw_results["ids"][0][i],
                    "source": source,
                }
            )
        return formatted

    def _find_knowledge_by_hash(self, user_id: int, fact_hash: str) -> dict | None:
        """Find an existing knowledge item for deduplication."""
        try:
            results = self.knowledge.get(
                where={"$and": [{"user_id": str(user_id)}, {"fact_hash": fact_hash}]}
            )
        except Exception as e:
            logger.debug("Knowledge dedupe lookup failed: %s", e)
            return None

        ids = results.get("ids", [])
        if not ids:
            return None

        metadatas = results.get("metadatas", [{}])
        documents = results.get("documents", [""])
        return {
            "id": ids[0],
            "document": documents[0] if documents else "",
            "metadata": metadatas[0] if metadatas else {},
        }

    def _increment_knowledge_access(self, memory: dict):
        """Increment access count for knowledge used in a prompt."""
        try:
            metadata = memory.get("metadata", {}).copy()
            metadata["access_count"] = int(metadata.get("access_count", 0)) + 1
            metadata["last_accessed"] = datetime.now(timezone.utc).isoformat()
            self.knowledge.update(ids=[memory["id"]], metadatas=[metadata])
        except Exception:
            logger.debug("Failed to update knowledge access count", exc_info=True)
