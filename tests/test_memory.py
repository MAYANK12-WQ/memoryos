"""
Unit tests for MemoryOS core functionality.
Run with: pytest tests/
"""

import math
import time
import pytest
import tempfile
import os

from memoryos.models import Memory, MemoryType, MemoryQuery
from memoryos.store import MemoryStore
from memoryos.agent import MemoryAgent


@pytest.fixture
def tmp_db(tmp_path):
    return str(tmp_path / "test.db")


@pytest.fixture
def store(tmp_db):
    return MemoryStore(db_path=tmp_db)


@pytest.fixture
def agent(tmp_path):
    return MemoryAgent(
        agent_id="test-agent",
        db_path=str(tmp_path / "agent.db"),
        vector_dir=str(tmp_path / "vectors"),
    )


# --- Model Tests ---

class TestMemoryModel:
    def test_retention_fresh_memory(self):
        m = Memory(content="test", memory_type=MemoryType.EPISODIC, agent_id="a")
        assert m.retention > 0.99  # just created, nearly full retention

    def test_retention_never_negative(self):
        m = Memory(content="test", memory_type=MemoryType.EPISODIC, agent_id="a")
        assert m.retention >= 0.0

    def test_reinforce_increases_stability(self):
        m = Memory(content="test", memory_type=MemoryType.EPISODIC, agent_id="a")
        initial_stability = m.stability
        m.reinforce()
        assert m.stability >= initial_stability

    def test_reinforce_increments_access_count(self):
        m = Memory(content="test", memory_type=MemoryType.EPISODIC, agent_id="a")
        m.reinforce()
        assert m.access_count == 1
        m.reinforce()
        assert m.access_count == 2

    def test_is_forgotten_fresh_memory(self):
        m = Memory(content="test", memory_type=MemoryType.EPISODIC, agent_id="a")
        assert not m.is_forgotten

    def test_to_dict_roundtrip(self):
        m = Memory(
            content="Hello world",
            memory_type=MemoryType.SEMANTIC,
            agent_id="agent-x",
            importance=8.5,
            tags=["tag1", "tag2"],
        )
        d = m.to_dict()
        assert d["content"] == "Hello world"
        assert d["memory_type"] == "semantic"
        assert d["importance"] == 8.5
        assert "retention" in d

    def test_ebbinghaus_formula(self):
        """Validate that retention follows e^(-t/S) roughly."""
        m = Memory(content="test", memory_type=MemoryType.EPISODIC, agent_id="a", stability=1.0)
        # Retention at t=0 should be ~1.0
        assert m.retention > 0.99


# --- Store Tests ---

class TestMemoryStore:
    def test_save_and_retrieve(self, store):
        m = Memory(content="store test", memory_type=MemoryType.EPISODIC, agent_id="agent1")
        store.save(m)
        fetched = store.get(m.id)
        assert fetched is not None
        assert fetched.content == "store test"

    def test_get_nonexistent(self, store):
        assert store.get("nonexistent-id") is None

    def test_get_all_filters_by_agent(self, store):
        m1 = Memory(content="agent1 memory", memory_type=MemoryType.EPISODIC, agent_id="agent1")
        m2 = Memory(content="agent2 memory", memory_type=MemoryType.EPISODIC, agent_id="agent2")
        store.save(m1)
        store.save(m2)

        agent1_memories = store.get_all("agent1")
        assert all(m.agent_id == "agent1" for m in agent1_memories)
        assert len(agent1_memories) == 1

    def test_delete_memory(self, store):
        m = Memory(content="to delete", memory_type=MemoryType.EPISODIC, agent_id="agent1")
        store.save(m)
        assert store.delete(m.id)
        assert store.get(m.id) is None

    def test_update_on_save(self, store):
        m = Memory(content="original", memory_type=MemoryType.EPISODIC, agent_id="agent1")
        store.save(m)
        m.content = "updated"
        store.save(m)
        fetched = store.get(m.id)
        assert fetched.content == "updated"

    def test_stats_empty(self, store):
        stats = store.stats("nonexistent-agent")
        assert stats["total"] == 0

    def test_stats_populated(self, store):
        for i in range(3):
            m = Memory(
                content=f"memory {i}",
                memory_type=MemoryType.EPISODIC,
                agent_id="stats-agent",
                importance=float(i + 1),
            )
            store.save(m)
        stats = store.stats("stats-agent")
        assert stats["total"] == 3
        assert "avg_retention" in stats

    def test_link_memories(self, store):
        m1 = Memory(content="memory A", memory_type=MemoryType.EPISODIC, agent_id="agent1")
        m2 = Memory(content="memory B", memory_type=MemoryType.EPISODIC, agent_id="agent1")
        store.save(m1)
        store.save(m2)
        store.link(m1.id, m2.id, "related_to", strength=0.9)
        linked = store.get_linked(m1.id)
        assert any(m.id == m2.id for m in linked)

    def test_tags_persist(self, store):
        m = Memory(
            content="tagged memory",
            memory_type=MemoryType.SEMANTIC,
            agent_id="agent1",
            tags=["python", "ai"],
        )
        store.save(m)
        fetched = store.get(m.id)
        assert fetched.tags == ["python", "ai"]


# --- Agent Tests ---

class TestMemoryAgent:
    def test_remember_and_recall(self, agent):
        agent.remember("User prefers dark mode", importance=7.0)
        results = agent.recall("What does the user prefer?", top_k=5)
        assert len(results) >= 1
        assert any("dark mode" in m.content for m in results)

    def test_recall_empty(self, agent):
        results = agent.recall("something not stored", top_k=5)
        assert isinstance(results, list)

    def test_recall_as_context_format(self, agent):
        agent.remember("User is named Mayank", importance=9.0, memory_type=MemoryType.SEMANTIC)
        ctx = agent.recall_as_context("who is the user?", top_k=3)
        assert "[AGENT MEMORY CONTEXT]" in ctx

    def test_forget(self, agent):
        m = agent.remember("temporary memory", importance=3.0)
        assert agent.forget(m.id)
        result = agent.store.get(m.id)
        assert result is None

    def test_stats_returns_dict(self, agent):
        agent.remember("test memory")
        stats = agent.stats()
        assert "total" in stats
        assert stats["total"] >= 1

    def test_dump(self, agent):
        agent.remember("dump test memory")
        dumped = agent.dump()
        assert isinstance(dumped, list)
        assert len(dumped) >= 1

    def test_memory_types_stored_correctly(self, agent):
        agent.remember("episodic fact", memory_type=MemoryType.EPISODIC)
        agent.remember("semantic fact", memory_type=MemoryType.SEMANTIC)
        agent.remember("procedural fact", memory_type=MemoryType.PROCEDURAL)

        all_mem = agent.store.get_all(agent.agent_id)
        types = {m.memory_type for m in all_mem}
        assert MemoryType.EPISODIC in types
        assert MemoryType.SEMANTIC in types
        assert MemoryType.PROCEDURAL in types

    def test_reinforcement_on_recall(self, agent):
        m = agent.remember("reinforcement test", importance=5.0)
        original_count = m.access_count

        agent.recall("reinforcement test", top_k=1)

        updated = agent.store.get(m.id)
        assert updated.access_count > original_count
