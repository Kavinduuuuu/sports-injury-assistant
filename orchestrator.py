"""
orchestrator.py
───────────────
Orchestrator — Agent-to-Agent Communication
============================================
This module wires the three agents together and manages the flow of
structured JSON messages between them. It is the "brain" that decides
the order of operations and what data each agent receives.

Pipeline flow:
    User Query
        │
        ▼
    TriageAgent  ──────────────────────────────► JSON triage_result
        │                                            │
        ▼                                            │
    RAGAgent  ◄──────────────────────────────────── ┘
    (receives query + triage_result)
        │
        ▼
    JSON rag_result
        │
        ▼
    ReflectionAgent
    (receives query + triage_result + rag_result)
        │
        ▼
    JSON final_result  ──────────────────────────► Streamlit UI

The messages between agents are plain Python dicts (JSON-serialisable).
This is the "custom lightweight protocol" — no LangGraph or CrewAI needed.
"""

import json
import time
from typing import Optional

from agents.triage_agent     import TriageAgent
from agents.rag_agent        import RAGAgent
from agents.reflection_agent import ReflectionAgent


# ─────────────────────────────────────────────────────────────────────────────
# Structured message type (just a dict with a known schema)
# ─────────────────────────────────────────────────────────────────────────────
def make_message(sender: str, recipient: str, payload: dict) -> dict:
    """
    Creates a structured inter-agent message.
    Think of this as an "envelope" wrapping the data.

    Args:
        sender    : Name of the sending agent (for logging)
        recipient : Name of the receiving agent (for logging)
        payload   : The actual data dict to pass along

    Returns a message dict:
        {
            "sender":    "TriageAgent",
            "recipient": "RAGAgent",
            "timestamp": 1722591234.567,
            "payload":   { ... }
        }
    """
    return {
        "sender":    sender,
        "recipient": recipient,
        "timestamp": time.time(),
        "payload":   payload,
    }


def log_message(msg: dict):
    """Prints a structured inter-agent message to the console for debugging."""
    print(
        f"\n{'-'*60}\n"
        f"msg: {msg['sender']} -> {msg['recipient']}\n"
        f"Payload: {json.dumps(msg['payload'], indent=2)[:500]}\n"
        f"{'-'*60}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Orchestrator class
# ─────────────────────────────────────────────────────────────────────────────
class Orchestrator:
    """
    Manages the sequential pipeline of three agents.

    The Orchestrator:
    - Instantiates all three agents once (on __init__)
    - Runs them in order, passing structured messages between them
    - Returns a complete result dict to the Streamlit UI
    - Captures timing info for each agent step

    Usage:
        orch   = Orchestrator()
        result = orch.run("I sprained my ankle during football.")
    """

    def __init__(self):
        print("[Orchestrator] Initialising agents...")
        self.triage_agent     = TriageAgent()
        self.rag_agent        = RAGAgent()
        self.reflection_agent = ReflectionAgent()
        print("[Orchestrator] All agents ready.")

    def run(self, user_query: str) -> dict:
        """
        Runs the full pipeline for a user query.

        Parameters
        ----------
        user_query : The user's free-text symptom description.

        Returns
        -------
        A result dict containing everything the UI needs:
        {
            "user_query":       str,
            "triage":           dict,   # from TriageAgent
            "rag":              dict,   # from RAGAgent (includes "chunks")
            "reflection":       dict,   # from ReflectionAgent
            "final_answer":     str,    # the ReflectionAgent's revised_answer
            "timings_ms":       dict,   # how long each agent took
            "error":            str | None
        }
        """
        print(f"\n{'-'*60}")
        print(f"[Orchestrator] New query: {user_query[:80]}...")
        print(f"{'-'*60}")

        timings = {}

        # ── STEP 1: TriageAgent ───────────────────────────────────────────────
        t0 = time.time()
        try:
            triage_result = self.triage_agent.run(user_query)
        except Exception as e:
            return self._error_result(user_query, f"TriageAgent failed: {e}")
        timings["triage_ms"] = int((time.time() - t0) * 1000)

        # Log the message being passed to RAGAgent
        msg_to_rag = make_message(
            sender    = "TriageAgent",
            recipient = "RAGAgent",
            payload   = {
                "user_query":    user_query,
                "triage_result": triage_result,
            },
        )
        log_message(msg_to_rag)

        # ── STEP 2: RAGAgent ─────────────────────────────────────────────────
        t0 = time.time()
        try:
            rag_result = self.rag_agent.run(
                user_query    = user_query,
                triage_result = triage_result,
            )
        except Exception as e:
            return self._error_result(user_query, f"RAGAgent failed: {e}",
                                      triage_result=triage_result)
        timings["rag_ms"] = int((time.time() - t0) * 1000)

        # Log the message being passed to ReflectionAgent
        msg_to_reflection = make_message(
            sender    = "RAGAgent",
            recipient = "ReflectionAgent",
            payload   = {
                "user_query":    user_query,
                "triage_result": triage_result,
                "rag_result":    {k: v for k, v in rag_result.items()
                                  if k != "chunks"},  # omit large chunk text
            },
        )
        log_message(msg_to_reflection)

        # ── STEP 3: ReflectionAgent ───────────────────────────────────────────
        t0 = time.time()
        try:
            reflection_result = self.reflection_agent.run(
                rag_result    = rag_result,
                triage_result = triage_result,
                user_query    = user_query,
            )
        except Exception as e:
            # If reflection fails, fall back to the unreviewed RAG answer
            print(f"[Orchestrator] [Warning] ReflectionAgent failed: {e}. Using raw RAG answer.")
            reflection_result = {
                "approved":       True,
                "revised_answer": rag_result.get("answer", ""),
                "notes":          f"Reflection skipped: {e}",
            }
        timings["reflection_ms"] = int((time.time() - t0) * 1000)

        # Log the final output message to the UI
        final_msg = make_message(
            sender    = "ReflectionAgent",
            recipient = "StreamlitUI",
            payload   = {
                "approved":     reflection_result["approved"],
                "final_answer": reflection_result["revised_answer"][:200] + "...",
            },
        )
        log_message(final_msg)

        # ── Compile final result ──────────────────────────────────────────────
        result = {
            "user_query":   user_query,
            "triage":       triage_result,
            "rag":          rag_result,
            "reflection":   reflection_result,
            "final_answer": reflection_result["revised_answer"],
            "timings_ms":   timings,
            "error":        None,
        }

        total_ms = sum(timings.values())
        print(
            f"\n[Orchestrator] Pipeline complete in {total_ms}ms "
            f"(triage: {timings['triage_ms']}ms, "
            f"rag: {timings['rag_ms']}ms, "
            f"reflection: {timings['reflection_ms']}ms)"
        )

        return result

    def _error_result(
        self,
        user_query: str,
        error_msg: str,
        triage_result: Optional[dict] = None,
    ) -> dict:
        """Returns a standardised error result dict."""
        print(f"[Orchestrator] [Warning] Error: {error_msg}")
        return {
            "user_query":   user_query,
            "triage":       triage_result or {},
            "rag":          {},
            "reflection":   {},
            "final_answer": "",
            "timings_ms":   {},
            "error":        error_msg,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Quick test (run: python orchestrator.py)
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    orch   = Orchestrator()
    result = orch.run("I sprained my ankle during a football match. It's swollen and sore.")
    print(f"\n{'═'*60}")
    print(f"FINAL ANSWER:\n{result['final_answer']}")
    print(f"\nTriage urgency: {result['triage'].get('urgency')}")
    print(f"Sources cited: {result['rag'].get('sources', [])}")
    print(f"Reflection approved: {result['reflection'].get('approved')}")
    print(f"Timings: {result['timings_ms']}")
