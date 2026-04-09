"""
Vector layer for semantic memory retrieval.
Uses ChromaDB with sentence-transformers embeddings.
Falls back to keyword search if ChromaDB is unavailable.
"""

import logging
from typing import Optional
from .models import Memory

logger = logging.getLogger(__name__)


class VectorMemoryIndex:
    """
    Manages semantic embeddings for memories using ChromaDB.
    Each agent gets its own isolated collection.
    """

    def __init__(self, persist_dir: str = ".memoryos_vectors"):
        self._persist_dir = persist_dir
        self._client = None
        self._encoder = None
        self._available = self._setup()

    def _setup(self) -> bool:
        try:
            import chromadb
            from chromadb.config import Settings

            self._client = chromadb.PersistentClient(
                path=self._persist_dir,
                settings=Settings(anonymized_telemetry=False),
            )
            logger.info("ChromaDB vector index initialized")
            return True
        except ImportError:
            logger.warning("chromadb not installed — falling back to keyword search")
            return False

    def _get_encoder(self):
        if self._encoder is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._encoder = SentenceTransformer("all-MiniLM-L6-v2")
            except ImportError:
                raise RuntimeError(
                    "sentence-transformers not installed. Run: pip install sentence-transformers"
                )
        return self._encoder

    def _collection(self, agent_id: str):
        safe_id = agent_id.replace("-", "_").replace(" ", "_")
        return self._client.get_or_create_collection(
            name=f"agent_{safe_id}",
            metadata={"hnsw:space": "cosine"},
        )

    def index(self, memory: Memory) -> str:
        """Embed and index a memory. Returns the embedding ID."""
        if not self._available:
            return memory.id

        encoder = self._get_encoder()
        embedding = encoder.encode(memory.content).tolist()
        collection = self._collection(memory.agent_id)

        collection.upsert(
            ids=[memory.id],
            embeddings=[embedding],
            documents=[memory.content],
            metadatas=[{
                "memory_type": memory.memory_type.value,
                "importance": memory.importance,
                "agent_id": memory.agent_id,
            }],
        )
        return memory.id

    def search(
        self,
        query: str,
        agent_id: str,
        top_k: int = 5,
        memory_type: Optional[str] = None,
    ) -> list[tuple[str, float]]:
        """
        Semantic search. Returns list of (memory_id, similarity_score) tuples.
        """
        if not self._available:
            return []

        encoder = self._get_encoder()
        query_embedding = encoder.encode(query).tolist()
        collection = self._collection(agent_id)

        where = {"agent_id": agent_id}
        if memory_type:
            where["memory_type"] = memory_type

        try:
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=min(top_k, collection.count() or 1),
                where=where if collection.count() > 0 else None,
            )
        except Exception as e:
            logger.warning(f"Vector search failed: {e}")
            return []

        ids = results["ids"][0] if results["ids"] else []
        distances = results["distances"][0] if results["distances"] else []

        # Convert cosine distance to similarity score
        return [(mid, round(1 - dist, 4)) for mid, dist in zip(ids, distances)]

    def delete(self, memory_id: str, agent_id: str) -> None:
        if not self._available:
            return
        try:
            self._collection(agent_id).delete(ids=[memory_id])
        except Exception as e:
            logger.warning(f"Failed to delete vector {memory_id}: {e}")

    @property
    def is_available(self) -> bool:
        return self._available
