"""
FastAPI REST interface for MemoryOS.
Allows any agent (LangChain, AutoGen, custom) to use MemoryOS over HTTP.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from memoryos import MemoryAgent
from memoryos.models import MemoryType

app = FastAPI(
    title="MemoryOS API",
    description="Persistent episodic + semantic memory with Ebbinghaus forgetting curves",
    version="0.1.0",
)

# In production, use a proper registry; for demo, single shared agent
_agents: dict[str, MemoryAgent] = {}


def _get_agent(agent_id: str) -> MemoryAgent:
    if agent_id not in _agents:
        _agents[agent_id] = MemoryAgent(agent_id=agent_id)
    return _agents[agent_id]


# --- Request/Response models ---

class RememberRequest(BaseModel):
    content: str
    memory_type: str = "episodic"
    importance: float = 5.0
    tags: list[str] = []
    source: str = "direct"

class RecallRequest(BaseModel):
    query: str
    top_k: int = 5
    memory_type: Optional[str] = None
    min_importance: float = 0.0
    include_forgotten: bool = False

class ContextRequest(BaseModel):
    query: str
    top_k: int = 5


# --- Endpoints ---

@app.get("/")
def root():
    return {"service": "MemoryOS", "version": "0.1.0", "status": "ok"}


@app.post("/agents/{agent_id}/remember")
def remember(agent_id: str, req: RememberRequest):
    agent = _get_agent(agent_id)
    try:
        mtype = MemoryType(req.memory_type)
    except ValueError:
        raise HTTPException(400, f"Invalid memory_type: {req.memory_type}")

    memory = agent.remember(
        content=req.content,
        memory_type=mtype,
        importance=req.importance,
        tags=req.tags,
        source=req.source,
    )
    return memory.to_dict()


@app.post("/agents/{agent_id}/recall")
def recall(agent_id: str, req: RecallRequest):
    agent = _get_agent(agent_id)
    mtype = MemoryType(req.memory_type) if req.memory_type else None
    memories = agent.recall(
        query=req.query,
        top_k=req.top_k,
        memory_type=mtype,
        min_importance=req.min_importance,
        include_forgotten=req.include_forgotten,
    )
    return [m.to_dict() for m in memories]


@app.post("/agents/{agent_id}/context")
def context(agent_id: str, req: ContextRequest):
    agent = _get_agent(agent_id)
    ctx = agent.recall_as_context(req.query, top_k=req.top_k)
    return {"context": ctx}


@app.get("/agents/{agent_id}/stats")
def stats(agent_id: str):
    return _get_agent(agent_id).stats()


@app.delete("/agents/{agent_id}/memories/{memory_id}")
def forget(agent_id: str, memory_id: str):
    agent = _get_agent(agent_id)
    deleted = agent.forget(memory_id)
    if not deleted:
        raise HTTPException(404, "Memory not found")
    return {"deleted": memory_id}


@app.post("/agents/{agent_id}/purge")
def purge(agent_id: str):
    count = _get_agent(agent_id).purge_forgotten()
    return {"purged": count}


@app.get("/agents/{agent_id}/dump")
def dump(agent_id: str):
    return _get_agent(agent_id).dump()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080, reload=True)
