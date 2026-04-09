"""
Smart memory retriever.
Combines semantic similarity + recency + importance + retention into a single relevance score.
This is the core intelligence of MemoryOS.
"""

import logging
from datetime import datetime
from typing import Optional

from .models import Memory, MemoryQuery, MemoryType
from .store import MemoryStore
from .vector import VectorMemoryIndex

logger = logging.getLogger(__name__)


def _recency_score(memory: Memory) -> float:
    """
    Recency bias: memories accessed within the last hour score 1.0,
    decaying exponentially over days.
    """
    import math
    hours_ago = (datetime.utcnow() - memory.last_accessed).total_seconds() / 3600
    return math.exp(-hours_ago / 48)  # half-life of 2 days


def _composite_score(
    memory: Memory,
    semantic_similarity: float,
    weights: dict,
) -> float:
    """
    Weighted composite relevance score.
    All components normalized to [0, 1].

    Score = w_sem * similarity
           + w_ret * retention
           + w_imp * (importance / 10)
           + w_rec * recency
    """
    importance_norm = memory.importance / 10.0
    recency = _recency_score(memory)
    retention = memory.retention

    return (
        weights["semantic"] * semantic_similarity
        + weights["retention"] * retention
        + weights["importance"] * importance_norm
        + weights["recency"] * recency
    )


class MemoryRetriever:
    """
    Retrieves the most relevant memories for a given query using a composite
    scoring function that balances semantic similarity, retention, importance,
    and recency.
    """

    DEFAULT_WEIGHTS = {
        "semantic": 0.50,
        "retention": 0.20,
        "importance": 0.20,
        "recency": 0.10,
    }

    def __init__(
        self,
        store: MemoryStore,
        vector_index: VectorMemoryIndex,
        weights: Optional[dict] = None,
    ):
        self.store = store
        self.vector = vector_index
        self.weights = weights or self.DEFAULT_WEIGHTS

    def retrieve(self, query: MemoryQuery) -> list[Memory]:
        """
        Retrieve top-k memories for the query.
        Reinforces retrieved memories (spaced repetition).
        """
        if self.vector.is_available:
            return self._semantic_retrieve(query)
        return self._keyword_retrieve(query)

    def _semantic_retrieve(self, query: MemoryQuery) -> list[Memory]:
        # Get semantic candidates — fetch more than needed to allow re-ranking
        candidates = self.vector.search(
            query=query.text,
            agent_id=query.agent_id,
            top_k=query.top_k * 3,
            memory_type=query.memory_type.value if query.memory_type else None,
        )

        sim_map = {mid: score for mid, score in candidates}

        # Load from DB and apply composite scoring
        scored: list[tuple[Memory, float]] = []
        for memory_id, sim in sim_map.items():
            memory = self.store.get(memory_id)
            if not memory:
                continue
            if memory.importance < query.min_importance:
                continue
            if not query.include_forgotten and memory.retention < query.min_retention:
                continue

            score = _composite_score(memory, sim, self.weights)
            scored.append((memory, score))

        # Sort by composite score descending
        scored.sort(key=lambda x: x[1], reverse=True)
        top = [m for m, _ in scored[: query.top_k]]

        # Reinforce retrieved memories
        for memory in top:
            self.store.update_after_recall(memory)

        return top

    def _keyword_retrieve(self, query: MemoryQuery) -> list[Memory]:
        """Fallback when ChromaDB is unavailable — simple keyword overlap scoring."""
        all_memories = self.store.get_all(
            agent_id=query.agent_id,
            memory_type=query.memory_type,
            min_importance=query.min_importance,
        )

        query_words = set(query.text.lower().split())
        scored: list[tuple[Memory, float]] = []

        for memory in all_memories:
            if not query.include_forgotten and memory.is_forgotten:
                continue
            if memory.retention < query.min_retention:
                continue

            memory_words = set(memory.content.lower().split())
            overlap = len(query_words & memory_words) / max(len(query_words), 1)
            score = _composite_score(memory, overlap, self.weights)
            scored.append((memory, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        top = [m for m, _ in scored[: query.top_k]]

        for memory in top:
            self.store.update_after_recall(memory)

        return top

    def retrieve_with_scores(self, query: MemoryQuery) -> list[tuple[Memory, float]]:
        """Same as retrieve() but returns scores alongside memories."""
        if self.vector.is_available:
            candidates = self.vector.search(
                query=query.text,
                agent_id=query.agent_id,
                top_k=query.top_k * 3,
            )
            sim_map = {mid: score for mid, score in candidates}

            scored = []
            for memory_id, sim in sim_map.items():
                memory = self.store.get(memory_id)
                if not memory:
                    continue
                if memory.retention < query.min_retention and not query.include_forgotten:
                    continue
                score = _composite_score(memory, sim, self.weights)
                scored.append((memory, score))

            scored.sort(key=lambda x: x[1], reverse=True)
            return scored[: query.top_k]

        return []
