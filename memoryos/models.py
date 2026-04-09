"""
Core data models for MemoryOS.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional
import uuid


class MemoryType(str, Enum):
    EPISODIC = "episodic"    # specific events: "User said they prefer async Python"
    SEMANTIC = "semantic"    # general facts: "User is an AI engineer"
    PROCEDURAL = "procedural"  # how-to knowledge: "User deploys via Docker"


@dataclass
class Memory:
    content: str
    memory_type: MemoryType
    agent_id: str
    importance: float = 5.0          # 1.0 (trivial) to 10.0 (critical)
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_accessed: datetime = field(default_factory=datetime.utcnow)
    access_count: int = 0
    stability: float = 1.0           # Ebbinghaus stability — increases on recall
    embedding_id: Optional[str] = None
    tags: list[str] = field(default_factory=list)
    source: str = "direct"           # direct | inferred | reinforced

    @property
    def retention(self) -> float:
        """
        Ebbinghaus retention formula: R = e^(-t/S)
        R = retrievability (0-1)
        t = time elapsed in days since last access
        S = stability (grows with each recall)
        """
        import math
        t = (datetime.utcnow() - self.last_accessed).total_seconds() / 86400
        return math.exp(-t / max(self.stability, 0.1))

    @property
    def is_forgotten(self) -> bool:
        """Memory is considered forgotten when retention drops below 5%."""
        return self.retention < 0.05

    def reinforce(self) -> None:
        """Called on successful recall — increases stability (spaced repetition effect)."""
        self.access_count += 1
        self.last_accessed = datetime.utcnow()
        # Stability grows logarithmically — each recall gives diminishing returns
        import math
        self.stability = self.stability * (1 + 0.3 / math.log(self.access_count + 2))

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "content": self.content,
            "memory_type": self.memory_type.value,
            "agent_id": self.agent_id,
            "importance": self.importance,
            "created_at": self.created_at.isoformat(),
            "last_accessed": self.last_accessed.isoformat(),
            "access_count": self.access_count,
            "stability": self.stability,
            "embedding_id": self.embedding_id,
            "tags": self.tags,
            "source": self.source,
            "retention": round(self.retention, 4),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Memory":
        return cls(
            id=d["id"],
            content=d["content"],
            memory_type=MemoryType(d["memory_type"]),
            agent_id=d["agent_id"],
            importance=d["importance"],
            created_at=datetime.fromisoformat(d["created_at"]),
            last_accessed=datetime.fromisoformat(d["last_accessed"]),
            access_count=d["access_count"],
            stability=d["stability"],
            embedding_id=d.get("embedding_id"),
            tags=d.get("tags", []),
            source=d.get("source", "direct"),
        )


@dataclass
class MemoryQuery:
    text: str
    agent_id: str
    top_k: int = 5
    memory_type: Optional[MemoryType] = None
    min_importance: float = 0.0
    min_retention: float = 0.05      # skip forgotten memories by default
    include_forgotten: bool = False
