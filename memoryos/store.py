"""
SQLite-backed episodic memory store.
Handles persistence, decay scheduling, and relational queries.
"""

import json
import logging
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Optional

from .models import Memory, MemoryType

logger = logging.getLogger(__name__)


SCHEMA = """
CREATE TABLE IF NOT EXISTS memories (
    id              TEXT PRIMARY KEY,
    agent_id        TEXT NOT NULL,
    content         TEXT NOT NULL,
    memory_type     TEXT NOT NULL,
    importance      REAL NOT NULL DEFAULT 5.0,
    created_at      TEXT NOT NULL,
    last_accessed   TEXT NOT NULL,
    access_count    INTEGER NOT NULL DEFAULT 0,
    stability       REAL NOT NULL DEFAULT 1.0,
    embedding_id    TEXT,
    tags            TEXT NOT NULL DEFAULT '[]',
    source          TEXT NOT NULL DEFAULT 'direct'
);

CREATE INDEX IF NOT EXISTS idx_agent ON memories(agent_id);
CREATE INDEX IF NOT EXISTS idx_type  ON memories(memory_type);
CREATE INDEX IF NOT EXISTS idx_importance ON memories(importance DESC);

CREATE TABLE IF NOT EXISTS memory_links (
    id          TEXT PRIMARY KEY,
    memory_a    TEXT NOT NULL REFERENCES memories(id),
    memory_b    TEXT NOT NULL REFERENCES memories(id),
    relationship TEXT NOT NULL,
    strength    REAL NOT NULL DEFAULT 1.0,
    created_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS sessions (
    id          TEXT PRIMARY KEY,
    agent_id    TEXT NOT NULL,
    started_at  TEXT NOT NULL,
    ended_at    TEXT,
    memory_count INTEGER NOT NULL DEFAULT 0
);
"""


class MemoryStore:
    """
    SQLite-backed store for episodic and semantic memories.
    Manages persistence, retrieval, decay scheduling, and relational links.
    """

    def __init__(self, db_path: str = "memoryos.db"):
        self.db_path = Path(db_path)
        self._init_db()

    def _init_db(self) -> None:
        with self._conn() as conn:
            conn.executescript(SCHEMA)
        logger.info(f"MemoryOS store initialized at {self.db_path}")

    @contextmanager
    def _conn(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def save(self, memory: Memory) -> Memory:
        """Persist a memory. Updates if ID already exists."""
        with self._conn() as conn:
            conn.execute(
                """
                INSERT INTO memories
                    (id, agent_id, content, memory_type, importance,
                     created_at, last_accessed, access_count, stability,
                     embedding_id, tags, source)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(id) DO UPDATE SET
                    content       = excluded.content,
                    importance    = excluded.importance,
                    last_accessed = excluded.last_accessed,
                    access_count  = excluded.access_count,
                    stability     = excluded.stability,
                    embedding_id  = excluded.embedding_id,
                    tags          = excluded.tags
                """,
                (
                    memory.id,
                    memory.agent_id,
                    memory.content,
                    memory.memory_type.value,
                    memory.importance,
                    memory.created_at.isoformat(),
                    memory.last_accessed.isoformat(),
                    memory.access_count,
                    memory.stability,
                    memory.embedding_id,
                    json.dumps(memory.tags),
                    memory.source,
                ),
            )
        return memory

    def get(self, memory_id: str) -> Optional[Memory]:
        """Fetch a single memory by ID."""
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM memories WHERE id = ?", (memory_id,)
            ).fetchone()
        return self._row_to_memory(row) if row else None

    def get_all(
        self,
        agent_id: str,
        memory_type: Optional[MemoryType] = None,
        min_importance: float = 0.0,
        limit: int = 100,
    ) -> list[Memory]:
        """Fetch all memories for an agent with optional filters."""
        query = "SELECT * FROM memories WHERE agent_id = ? AND importance >= ?"
        params: list = [agent_id, min_importance]

        if memory_type:
            query += " AND memory_type = ?"
            params.append(memory_type.value)

        query += " ORDER BY importance DESC, last_accessed DESC LIMIT ?"
        params.append(limit)

        with self._conn() as conn:
            rows = conn.execute(query, params).fetchall()
        return [self._row_to_memory(r) for r in rows]

    def update_after_recall(self, memory: Memory) -> None:
        """Reinforce memory after successful retrieval."""
        memory.reinforce()
        self.save(memory)

    def delete(self, memory_id: str) -> bool:
        """Delete a memory by ID."""
        with self._conn() as conn:
            affected = conn.execute(
                "DELETE FROM memories WHERE id = ?", (memory_id,)
            ).rowcount
        return affected > 0

    def purge_forgotten(self, agent_id: str) -> int:
        """
        Remove memories whose retention has dropped below threshold.
        Returns count of purged memories.
        """
        all_memories = self.get_all(agent_id, limit=10000)
        forgotten = [m for m in all_memories if m.is_forgotten]
        with self._conn() as conn:
            for m in forgotten:
                conn.execute("DELETE FROM memories WHERE id = ?", (m.id,))
        logger.info(f"Purged {len(forgotten)} forgotten memories for agent {agent_id}")
        return len(forgotten)

    def link(
        self, memory_a_id: str, memory_b_id: str, relationship: str, strength: float = 1.0
    ) -> None:
        """Create a relational link between two memories."""
        import uuid
        with self._conn() as conn:
            conn.execute(
                """
                INSERT INTO memory_links (id, memory_a, memory_b, relationship, strength, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (str(uuid.uuid4()), memory_a_id, memory_b_id, relationship, strength, datetime.utcnow().isoformat()),
            )

    def get_linked(self, memory_id: str) -> list[Memory]:
        """Fetch all memories linked to a given memory."""
        with self._conn() as conn:
            rows = conn.execute(
                """
                SELECT m.* FROM memories m
                JOIN memory_links l ON (l.memory_b = m.id OR l.memory_a = m.id)
                WHERE (l.memory_a = ? OR l.memory_b = ?) AND m.id != ?
                ORDER BY l.strength DESC
                """,
                (memory_id, memory_id, memory_id),
            ).fetchall()
        return [self._row_to_memory(r) for r in rows]

    def stats(self, agent_id: str) -> dict:
        """Return memory health statistics for an agent."""
        all_memories = self.get_all(agent_id, limit=10000)
        if not all_memories:
            return {"total": 0}

        retentions = [m.retention for m in all_memories]
        return {
            "total": len(all_memories),
            "by_type": {
                t.value: sum(1 for m in all_memories if m.memory_type == t)
                for t in MemoryType
            },
            "avg_retention": round(sum(retentions) / len(retentions), 3),
            "forgotten": sum(1 for m in all_memories if m.is_forgotten),
            "avg_importance": round(sum(m.importance for m in all_memories) / len(all_memories), 2),
            "most_stable": max(all_memories, key=lambda m: m.stability).content[:60],
        }

    @staticmethod
    def _row_to_memory(row: sqlite3.Row) -> Memory:
        return Memory(
            id=row["id"],
            content=row["content"],
            memory_type=MemoryType(row["memory_type"]),
            agent_id=row["agent_id"],
            importance=row["importance"],
            created_at=datetime.fromisoformat(row["created_at"]),
            last_accessed=datetime.fromisoformat(row["last_accessed"]),
            access_count=row["access_count"],
            stability=row["stability"],
            embedding_id=row["embedding_id"],
            tags=json.loads(row["tags"]),
            source=row["source"],
        )
