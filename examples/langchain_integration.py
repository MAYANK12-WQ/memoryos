"""
MemoryOS + LangChain Integration Example
Shows how to use MemoryOS as a persistent memory backend for a LangChain conversational agent.
Requires: pip install langchain langchain-openai
"""

import json
from memoryos import MemoryAgent
from memoryos.models import MemoryType


class MemoryOSChatHistory:
    """
    Drop-in LangChain-compatible chat memory using MemoryOS as backend.
    Unlike ConversationBufferMemory, this persists across sessions and forgets intelligently.
    """

    def __init__(self, agent_id: str, llm=None):
        self.agent = MemoryAgent(agent_id=agent_id)
        self.llm = llm

    def save_context(self, inputs: dict, outputs: dict) -> None:
        """Called by LangChain after each exchange."""
        human_msg = inputs.get("input", "")
        ai_msg = outputs.get("output", "")

        self.agent.remember(
            f"Human: {human_msg}\nAssistant: {ai_msg}",
            memory_type=MemoryType.EPISODIC,
            importance=4.0,
            source="conversation",
        )

        if self.llm and human_msg:
            self._extract_and_store_facts(human_msg, ai_msg)

    def _extract_and_store_facts(self, human: str, ai: str) -> None:
        """Use LLM to extract durable semantic facts from the conversation."""
        try:
            prompt = (
                "Extract 0-2 important facts about the user from this exchange.\n"
                "Return ONLY a JSON array of strings. If no important facts, return [].\n\n"
                f"Human: {human}\nAssistant: {ai}\n\nFacts:"
            )
            response = self.llm.invoke(prompt)
            facts = json.loads(response.content if hasattr(response, "content") else response)
            for fact in facts:
                self.agent.remember(
                    fact,
                    memory_type=MemoryType.SEMANTIC,
                    importance=7.0,
                    source="inferred",
                )
        except Exception:
            pass  # Never let memory extraction break the main conversation

    def load_memory_variables(self, inputs: dict) -> dict:
        """Inject relevant memories into the LangChain prompt context."""
        query = inputs.get("input", "")
        context = self.agent.recall_as_context(query, top_k=5)
        return {"memory_context": context}

    @property
    def memory_variables(self) -> list[str]:
        return ["memory_context"]


def demo_without_llm():
    """
    Demo of MemoryOS chat history without a real LLM.
    Shows how context is built up across turns.
    """
    print("=== MemoryOS + LangChain Demo (no LLM required) ===\n")

    memory = MemoryOSChatHistory(agent_id="langchain-demo")

    # Simulate a multi-turn conversation
    exchanges = [
        ("I'm Mayank, an AI engineer from India", "Nice to meet you Mayank! How can I help?"),
        ("I mostly work with Python and LangChain for RAG systems", "Great stack! RAG with LangChain is very powerful."),
        ("I'm having trouble with async embedding calls timing out", "Let's debug that — are you using asyncio.gather?"),
        ("Yes, I'm using asyncio but the Pinecone client isn't async native", "You can wrap sync calls with asyncio.to_thread() for true async."),
    ]

    for human, ai in exchanges:
        memory.save_context({"input": human}, {"output": ai})
        print(f"Human: {human}")
        print(f"AI:    {ai}\n")

    # Now recall what MemoryOS knows
    print("\n--- What MemoryOS remembers about 'async Python issues' ---")
    ctx = memory.load_memory_variables({"input": "async Python issues"})
    print(ctx["memory_context"])

    print("\n--- Memory stats ---")
    stats = memory.agent.stats()
    for k, v in stats.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    demo_without_llm()
