# 🧠 MemoryOS

> **Persistent episodic + semantic memory for AI agents — with Ebbinghaus forgetting curves.**

Most AI agents start every session from zero. MemoryOS gives them a real memory — one that persists across restarts, retrieves what's *relevant* (not just what's *recent*), and intelligently forgets low-value memories over time using the same mathematics as human forgetting.

---

## The Problem

Every major agent framework has the same memory flaw:

| Approach | Problem |
|---|---|
| `ConversationBufferMemory` | Floods the context window. Forgets nothing. Crashes on long sessions. |
| `ConversationSummaryMemory` | Lossy. Summaries destroy nuance. |
| `VectorStoreRetriever` | No sense of time. A 3-month-old irrelevant fact scores the same as a fresh critical one. |
| **MemoryOS** | Persistent. Decaying. Semantically ranked. Reinforced on recall. |

---

## How It Works

```
New Memory → [SQLite Store] + [ChromaDB Vector Index]
                   ↓
           Ebbinghaus Decay Timer
           R = e^(-t/S)
           (R=retention, t=days, S=stability)
                   ↓
Query → Semantic Search + Composite Ranking
        score = 0.5*similarity + 0.2*retention + 0.2*importance + 0.1*recency
                   ↓
        Top-K Memories → Reinforce (spaced repetition)
```

### Ebbinghaus Forgetting Curve

The retention formula `R = e^(-t/S)` means:
- A memory with **stability=1** loses half its retention in ~0.7 days
- Every time a memory is **recalled**, stability grows (spaced repetition)
- Frequently recalled memories become increasingly permanent
- Unused, low-importance memories decay and are eventually purged

This mirrors how human long-term memory actually works.

---

## Architecture

```
memoryos/
├── models.py       # Memory dataclass with Ebbinghaus decay math
├── store.py        # SQLite persistence layer (WAL mode, indexed)
├── vector.py       # ChromaDB semantic index (sentence-transformers)
├── retriever.py    # Composite ranking: similarity + retention + importance + recency
├── agent.py        # High-level API (remember / recall / forget / stats)
api/
└── server.py       # FastAPI REST interface — use from any language/framework
examples/
├── basic_usage.py
└── langchain_integration.py
tests/
└── test_memory.py
```

---

## Quickstart

```bash
git clone https://github.com/MAYANK12WQ/memoryos.git
cd memoryos
pip install -r requirements.txt
python examples/basic_usage.py
```

### Basic Usage

```python
from memoryos import MemoryAgent
from memoryos.models import MemoryType

agent = MemoryAgent(agent_id="my-bot", db_path="my_bot.db")

# Store memories
agent.remember(
    "User prefers Python for all backend work",
    memory_type=MemoryType.SEMANTIC,
    importance=8.0,
    tags=["preference", "coding"],
)

agent.remember(
    "User deployed RAG pipeline using Pinecone last week",
    memory_type=MemoryType.EPISODIC,
    importance=7.0,
)

# Retrieve — automatically reinforces recalled memories
memories = agent.recall("What stack does the user prefer?", top_k=3)
for m in memories:
    print(f"[{m.memory_type.value}] retention={m.retention:.0%}  {m.content}")

# Inject into LLM system prompt
context = agent.recall_as_context("Tell me about the user")
print(context)
# [AGENT MEMORY CONTEXT]
# 1. [SEMANTIC] (importance=8.0, retention=99%)
#    User prefers Python for all backend work
# 2. [EPISODIC] (importance=7.0, retention=97%)
#    User deployed RAG pipeline using Pinecone last week
```

### Memory Types

| Type | Use For | Example |
|---|---|---|
| `EPISODIC` | Specific events | "User reported a bug on Tuesday" |
| `SEMANTIC` | General facts | "User is an AI engineer" |
| `PROCEDURAL` | How-to knowledge | "Always use async for API calls" |

### Importance Scale

| Score | Meaning |
|---|---|
| 1-3 | Trivial — decays fast, purged first |
| 4-6 | Normal context |
| 7-8 | Important — survives longer |
| 9-10 | Critical — nearly permanent |

---

## Memory Decay Simulation

```python
import math

def retention(stability, days_since_access):
    return math.exp(-days_since_access / stability)

# Importance=5 memory, stability=1.0
print(f"1 day:   {retention(1.0, 1):.0%}")   # 37%
print(f"2 days:  {retention(1.0, 2):.0%}")   # 14%
print(f"7 days:  {retention(1.0, 7):.0%}")   # <0.1%

# After 3 recalls (stability grows to ~1.6)
print(f"1 day:   {retention(1.6, 1):.0%}")   # 54%
print(f"7 days:  {retention(1.6, 7):.0%}")   # 1.3%
```

---

## REST API

```bash
uvicorn api.server:app --reload --port 8080
```

```bash
# Store a memory
curl -X POST http://localhost:8080/agents/my-bot/remember \
  -H "Content-Type: application/json" \
  -d '{"content": "User prefers dark mode", "importance": 7.0}'

# Recall
curl -X POST http://localhost:8080/agents/my-bot/recall \
  -H "Content-Type: application/json" \
  -d '{"query": "what does the user prefer?", "top_k": 3}'

# Get memory health stats
curl http://localhost:8080/agents/my-bot/stats

# Purge forgotten memories
curl -X POST http://localhost:8080/agents/my-bot/purge
```

---

## LangChain Integration

```python
from examples.langchain_integration import MemoryOSChatHistory

memory = MemoryOSChatHistory(agent_id="my-chatbot", llm=your_llm)

# Use exactly like ConversationBufferMemory — but it persists and forgets intelligently
chain = ConversationChain(llm=your_llm, memory=memory)
```

---

## Running Tests

```bash
pytest tests/ -v
```

---

## Database Schema

```sql
-- Core memory table (SQLite WAL mode)
CREATE TABLE memories (
    id            TEXT PRIMARY KEY,
    agent_id      TEXT NOT NULL,
    content       TEXT NOT NULL,
    memory_type   TEXT NOT NULL,      -- episodic | semantic | procedural
    importance    REAL DEFAULT 5.0,   -- 1.0-10.0
    created_at    TEXT NOT NULL,
    last_accessed TEXT NOT NULL,
    access_count  INTEGER DEFAULT 0,
    stability     REAL DEFAULT 1.0,   -- grows on recall (spaced repetition)
    embedding_id  TEXT,               -- ChromaDB vector reference
    tags          TEXT DEFAULT '[]',
    source        TEXT DEFAULT 'direct'
);

-- Relational memory links (associative memory)
CREATE TABLE memory_links (
    id           TEXT PRIMARY KEY,
    memory_a     TEXT REFERENCES memories(id),
    memory_b     TEXT REFERENCES memories(id),
    relationship TEXT NOT NULL,
    strength     REAL DEFAULT 1.0
);
```

---

## Composite Retrieval Score

```
score = 0.50 × semantic_similarity   (ChromaDB cosine distance)
      + 0.20 × retention             (Ebbinghaus R = e^(-t/S))
      + 0.20 × importance / 10       (user-defined 1-10 scale)
      + 0.10 × recency               (exponential decay, half-life 2 days)
```

Weights are configurable:
```python
agent = MemoryAgent(
    agent_id="my-bot",
    retrieval_weights={
        "semantic": 0.6,
        "retention": 0.15,
        "importance": 0.15,
        "recency": 0.1,
    }
)
```

---

## Roadmap

- [ ] Async SQLite support (aiosqlite)
- [ ] Qdrant backend option (for production scale)
- [ ] Memory compression (auto-summarize old episodic → semantic)
- [ ] Multi-agent shared memory with access control
- [ ] OpenTelemetry tracing for memory operations
- [ ] AutoGen / CrewAI integration examples

---

## License

MIT © Mayank Shekhar

---

> Built to solve a real gap in agent memory. If this helped you, leave a ⭐
