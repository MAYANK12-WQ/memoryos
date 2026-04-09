"""
MemoryOS — Basic Usage Example
Demonstrates core remember/recall/decay functionality.
"""

import time
from memoryos import MemoryAgent
from memoryos.models import MemoryType


def main():
    print("=== MemoryOS Basic Usage Demo ===\n")

    # Initialize agent — persistent across sessions
    agent = MemoryAgent(
        agent_id="demo-agent",
        db_path="demo.db",
        vector_dir=".demo_vectors",
    )

    # --- Store different types of memories ---
    print("Storing memories...")

    m1 = agent.remember(
        "User prefers Python over JavaScript for backend work",
        memory_type=MemoryType.SEMANTIC,
        importance=8.0,
        tags=["preference", "coding"],
    )

    m2 = agent.remember(
        "User deployed a RAG pipeline using Pinecone and LangChain on AWS last week",
        memory_type=MemoryType.EPISODIC,
        importance=7.5,
        tags=["deployment", "rag"],
    )

    m3 = agent.remember(
        "Always use async/await when calling external APIs to avoid blocking",
        memory_type=MemoryType.PROCEDURAL,
        importance=6.0,
        tags=["best-practice", "async"],
    )

    m4 = agent.remember(
        "User's name is Mayank and they are based in India",
        memory_type=MemoryType.SEMANTIC,
        importance=9.0,
        tags=["identity"],
    )

    # Link related memories
    agent.link_memories(m2.id, m1.id, relationship="uses_preferred_language", strength=0.8)
    print(f"Stored {4} memories\n")

    # --- Recall ---
    print("Recalling: 'What does the user prefer for coding?'")
    results = agent.recall("What does the user prefer for coding?", top_k=3)
    for i, mem in enumerate(results, 1):
        print(f"  {i}. [{mem.memory_type.value}] (retention={mem.retention:.0%}) {mem.content[:80]}")

    # --- Context for LLM ---
    print("\n--- LLM Context String ---")
    ctx = agent.recall_as_context("Tell me about the user's recent work")
    print(ctx)

    # --- Stats ---
    print("\n--- Memory Health Stats ---")
    stats = agent.stats()
    for k, v in stats.items():
        print(f"  {k}: {v}")

    # --- Decay simulation ---
    print("\n--- Retention Values ---")
    all_memories = agent.store.get_all("demo-agent")
    for m in all_memories:
        print(f"  [{m.memory_type.value:10s}] retention={m.retention:.2%}  stability={m.stability:.2f}  '{m.content[:50]}...'")

    print("\nDone. Database saved to demo.db")


if __name__ == "__main__":
    main()
