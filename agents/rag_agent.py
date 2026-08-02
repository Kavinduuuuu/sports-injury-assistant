"""
agents/rag_agent.py
───────────────────
RAGAgent — Tool-use / RAG Pattern
==================================
The second agent in the pipeline. It:

  1. Calls the retriever as a "tool" to fetch relevant knowledge-base chunks.
  2. Passes those chunks + the original query to a higher-reasoning LLM.
  3. The LLM synthesizes a grounded answer that cites the retrieved sources.

Design pattern: Tool-use + RAG — the agent treats retrieval as a callable
tool, then uses an LLM to reason over the retrieved evidence.

Model: OpenRouter API with a higher-reasoning model (e.g. Claude 3 Haiku).
       OpenRouter uses the same REST interface as the OpenAI Python SDK.
"""

import os
import json
from dotenv import load_dotenv

load_dotenv()

from openai import OpenAI  # OpenRouter is OpenAI-compatible

# Import our local retriever "tool"
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from rag.retriever import retrieve

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_MODEL   = os.getenv("OPENROUTER_MODEL", "anthropic/claude-3-haiku")
RAG_TOP_K          = int(os.getenv("RAG_TOP_K", "4"))

# OpenRouter's base URL (drop-in replacement for OpenAI's endpoint)
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

SYSTEM_PROMPT = """You are a knowledgeable sports injury recovery assistant.
You have been given a set of relevant excerpts from trusted sports medicine
and physiotherapy reference documents. Use ONLY these excerpts to answer
the user's question. Do NOT invent information that isn't in the excerpts.

When you cite information, reference the source excerpt by its number
(e.g. "According to excerpt [1]…").

IMPORTANT: Respond ONLY with valid JSON in this exact format:
{
  "answer": "your detailed, helpful answer here",
  "sources": ["chunk_id_1", "chunk_id_2"]
}
The "sources" list must contain only the chunk IDs of excerpts you actually used.
Do NOT include any text before or after the JSON object."""


# ─────────────────────────────────────────────────────────────────────────────
# Agent class
# ─────────────────────────────────────────────────────────────────────────────
class RAGAgent:
    """
    Retrieves relevant knowledge-base chunks and synthesizes a cited answer.

    Usage:
        agent  = RAGAgent()
        result = agent.run(
            user_query    = "How do I treat a sprained ankle?",
            triage_result = {"urgency": "self_manageable", "reasoning": "..."}
        )
        # result → {"answer": "...", "sources": ["abc123_0", "def456_1"]}
    """

    def __init__(self):
        if not OPENROUTER_API_KEY:
            raise ValueError(
                "OPENROUTER_API_KEY is not set. "
                "Add it to your .env file and restart."
            )
        # The OpenAI SDK works with OpenRouter by just changing base_url
        self.client = OpenAI(
            api_key  = OPENROUTER_API_KEY,
            base_url = OPENROUTER_BASE_URL,
        )
        self.model = OPENROUTER_MODEL

    def run(self, user_query: str, triage_result: dict) -> dict:
        """
        Retrieves chunks and synthesizes a grounded answer.

        Parameters
        ----------
        user_query    : The user's original symptom description.
        triage_result : Output from TriageAgent (passed in for context).

        Returns
        -------
        dict with keys:
            "answer"  : str (the synthesized response)
            "sources" : list[str] (chunk IDs of excerpts used)
            "chunks"  : list[dict] (the raw retrieved chunks, for the UI)

        On error, returns a meaningful error message.
        """
        print(f"\n[RAGAgent] Query: {user_query[:80]}...")
        print(f"[RAGAgent] Triage urgency: {triage_result.get('urgency')}")

        # ── TOOL CALL: Retrieve relevant chunks ───────────────────────────────
        # This is what makes it a "tool-use" pattern:
        # the agent decides to call the retrieval tool, then uses the result.
        chunks = retrieve(user_query, top_k=RAG_TOP_K)
        print(f"[RAGAgent] Retrieved {len(chunks)} chunks from knowledge base.")

        if not chunks:
            return {
                "answer":  (
                    "I wasn't able to find relevant information in the knowledge base. "
                    "Please ensure documents have been ingested (run: python rag/ingest.py) "
                    "and try a more specific query."
                ),
                "sources": [],
                "chunks":  [],
            }

        # ── Build context string from retrieved chunks ────────────────────────
        context_parts = []
        for i, chunk in enumerate(chunks, start=1):
            context_parts.append(
                f"[Excerpt {i}] (ID: {chunk['chunk_id']}, Source: {chunk['source']})\n"
                f"{chunk['text']}"
            )
        context_str = "\n\n---\n\n".join(context_parts)

        # ── Build the prompt ──────────────────────────────────────────────────
        user_message = (
            f"User question: {user_query}\n\n"
            f"Triage classification: {triage_result.get('urgency', 'unknown')} "
            f"— {triage_result.get('reasoning', '')}\n\n"
            f"Relevant excerpts from the knowledge base:\n\n{context_str}\n\n"
            f"Please answer the user's question based on these excerpts."
        )

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_message},
        ]

        # ── Call the OpenRouter API ───────────────────────────────────────────
        try:
            response = self.client.chat.completions.create(
                model       = self.model,
                messages    = messages,
                temperature = 0.2,   # low temperature for factual grounding
                max_tokens  = 1024,
            )
            raw_output = response.choices[0].message.content.strip()
            print(f"[RAGAgent] Raw output (first 200 chars): {raw_output[:200]}...")

        except Exception as e:
            print(f"[RAGAgent] [Warning] API error: {e}")
            return {
                "answer":  f"An error occurred while generating the answer: {e}",
                "sources": [],
                "chunks":  chunks,
            }

        # ── Parse the JSON response ───────────────────────────────────────────
        result = self._parse_output(raw_output)
        # Attach the raw chunks so the UI can display them
        result["chunks"] = chunks
        return result

    def _parse_output(self, raw_output: str) -> dict:
        """Safely parses the LLM's JSON response."""
        try:
            result = json.loads(raw_output)
            if "answer" not in result:
                raise ValueError("Missing 'answer' field.")
            if "sources" not in result:
                result["sources"] = []
            return result

        except (json.JSONDecodeError, ValueError) as e:
            print(f"[RAGAgent] [Warning] JSON parse error: {e}")
            # If JSON parsing fails, wrap the raw text as the answer
            return {
                "answer":  raw_output,
                "sources": [],
            }


# ─────────────────────────────────────────────────────────────────────────────
# Quick test (run: python agents/rag_agent.py)
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    agent  = RAGAgent()
    result = agent.run(
        user_query    = "What is the RICE protocol for ankle sprains?",
        triage_result = {"urgency": "self_manageable", "reasoning": "Minor sprain."},
    )
    print(f"\nAnswer:\n{result['answer']}")
    print(f"\nSources: {result['sources']}")
