"""
MemoryAgent — High-level API that ties store + retriever + vector index together.
This is what end-users and LangChain integrations interact with.
"""

import logging
from typing import Optional

from .models import Memory, MemoryQuery, MemoryType
from .store import MemoryStore
from .vector import VectorMemoryIndex
from .retriever import MemoryRetriever

logger = logging.getLogger(__name__)


class MemoryAgent:
    """
    The main interface to MemoryOS.

    Usage:
        agent = MemoryAgent(agent_id="my-bot")
        agent.remember("User prefers Python over JavaScript", importance=7.0)
        memories = agent.recall("What does the user prefer for coding?")
    """

    def __init__(
        self,
        agent_id: str,
        db_path: str = "memoryos.db",
        vector_dir: str = ".memoryos_vectors",
        retrieval_weights: Optional[dict] = None,
    ):
        self.agent_id = agent_id
        self.store = MemoryStore(db_path=db_path)
        self.vector = VectorMemoryIndex(persist_dir=vector_dir)
        self.retriever = MemoryRetriever(
            store=self.store,
            vector_index=self.vector,
            weights=retrieval_weights,
        )

    def remember(
        self,
        content: str,
        memory_type: MemoryType = MemoryType.EPISODIC,
        importance: float = 5.0,
        tags: Optional[list[str]] = None,
        source: str = "direct",
    ) -> Memory:
        """
        Store a new memory.

        Args:
            content: The memory text.
            memory_type: EPISODIC | SEMANTIC | PROCEDURAL
            importance: 1.0 (trivial) to 10.0 (critical)
            tags: Optional list of string labels for filtering.
            source: Origin — 'direct', 'inferred', 'reinforced'

        Returns:
            The stored Memory object.
        """
        memory = Memory(
            content=content,
            memory_type=memory_type,
            agent_id=self.agent_id,
            importance=importance,
            tags=tags or [],
            source=source,
        )

        # Index in vector store first to get embedding_id
        embedding_id = self.vector.index(memory)
        memory.embedding_id = embedding_id

        # Persist in SQLite
        self.store.save(memory)
        logger.info(f"Stored memory [{memory.memory_type.value}] id={memory.id[:8]}…")
        return memory

    def recall(
        self,
        query: str,
        top_k: int = 5,
        memory_type: Optional[MemoryType] = None,
        min_importance: float = 0.0,
        include_forgotten: bool = False,
    ) -> list[Memory]:
        """
        Retrieve the most relevant memories for a query.
        Automatically reinforces retrieved memories (spaced repetition).

        Args:
            query: Natural language query.
            top_k: Number of memories to return.
            memory_type: Filter by type (optional).
            min_importance: Only return memories above this threshold.
            include_forgotten: If True, return decayed memories too.

        Returns:
            List of Memory objects, ranked by composite relevance score.
        """
        q = MemoryQuery(
            text=query,
            agent_id=self.agent_id,
            top_k=top_k,
            memory_type=memory_type,
            min_importance=min_importance,
            include_forgotten=include_forgotten,
        )
        return self.retriever.retrieve(q)

    def recall_as_context(self, query: str, top_k: int = 5) -> str:
        """
        Retrieve memories and format them as a context string for LLM prompts.

        Returns:
            A formatted string ready to inject into a system prompt.
        """
        memories = self.recall(query, top_k=top_k)
        if not memories:
            return "No relevant memories found."

        lines = ["[AGENT MEMORY CONTEXT]"]
        for i, m in enumerate(memories, 1):
            lines.append(
                f"{i}. [{m.memory_type.value.upper()}] (importance={m.importance:.1f}, "
                f"retention={m.retention:.0%})\n   {m.content}"
            )
        return "\n".join(lines)

    def forget(self, memory_id: str) -> bool:
        """Explicitly delete a memory."""
        self.vector.delete(memory_id, self.agent_id)
        return self.store.delete(memory_id)

    def purge_forgotten(self) -> int:
        """Remove all memories that have decayed below retention threshold."""
        return self.store.purge_forgotten(self.agent_id)

    def link_memories(
        self, memory_a_id: str, memory_b_id: str, relationship: str, strength: float = 1.0
    ) -> None:
        """Create an explicit relational link between two memories."""
        self.store.link(memory_a_id, memory_b_id, relationship, strength)

    def get_linked(self, memory_id: str) -> list[Memory]:
        """Get all memories linked to a given memory."""
        return self.store.get_linked(memory_id)

    def stats(self) -> dict:
        """Return memory health statistics."""
        s = self.store.stats(self.agent_id)
        s["vector_index_available"] = self.vector.is_available
        s["agent_id"] = self.agent_id
        return s

    def dump(self) -> list[dict]:
        """Export all memories as a list of dicts."""
        return [m.to_dict() for m in self.store.get_all(self.agent_id, limit=100000)]

    # LangChain compatibility
    def add_texts(self, texts: list[str], **kwargs) -> list[str]:
        """LangChain Memory interface compatibility."""
        ids = []
        for text in texts:
            m = self.remember(text, source="langchain")
            ids.append(m.id)
        return ids

    def search(self, query: str, k: int = 4) -> list[str]:
        """LangChain VectorStore-like interface."""
        memories = self.recall(query, top_k=k)
        return [m.content for m in memories]
