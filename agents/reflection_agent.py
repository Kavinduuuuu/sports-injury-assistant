"""
agents/reflection_agent.py
──────────────────────────
ReflectionAgent — Reflection / Self-Critique Pattern
=====================================================
The third and final agent in the pipeline. It acts as a quality-control
layer that reviews the RAGAgent's answer before it reaches the user.

It checks:
  1. Medical responsibility — does the answer avoid overconfident claims?
  2. Consistency with triage — if triage said "see_doctor_immediately",
     does the answer clearly recommend seeing a doctor?
  3. Safety — does the answer avoid diagnosing, prescribing, or replacing
     professional medical advice?

If the answer passes all checks → approved as-is.
If not → the agent produces a revised, safer answer.

Design pattern: Reflection / Self-Critique — an agent reviews and
optionally corrects another agent's output.

Model: Groq + Llama 3.1 8B (fast; reflection needs common sense, not depth)
"""

import os
import json
from dotenv import load_dotenv

load_dotenv()

from groq import Groq

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL   = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")

SYSTEM_PROMPT = """You are a medical safety reviewer for a sports injury AI assistant.
Your job is to review an AI-generated answer about a sports injury and check it for safety.

Review criteria:
1. DOCTOR REFERRAL CHECK: If the triage urgency is "see_doctor_immediately", the answer
   MUST clearly tell the user to see a doctor or go to emergency. If it doesn't, revise it.
2. OVERCONFIDENCE CHECK: The answer must NOT:
   - Diagnose a specific condition (e.g. "You have a torn ACL")
   - Prescribe specific medications or dosages
   - Guarantee recovery timelines as fact
3. DISCLAIMER CHECK: The answer should acknowledge it's general advice, not a replacement
   for professional medical assessment.
4. ACCURACY CHECK: The answer should be sensible and medically reasonable.

IMPORTANT: Respond ONLY with valid JSON in this exact format:
{
  "approved": true or false,
  "revised_answer": "the revised or original answer text",
  "notes": "brief explanation of what (if anything) you changed and why"
}

If the answer is good, set "approved" to true and copy the original answer
into "revised_answer" unchanged.
If you made changes, set "approved" to false and put the improved version
in "revised_answer".
Do NOT include any text before or after the JSON object."""


# ─────────────────────────────────────────────────────────────────────────────
# Agent class
# ─────────────────────────────────────────────────────────────────────────────
class ReflectionAgent:
    """
    Reviews and optionally corrects the RAGAgent's answer for medical safety.

    Usage:
        agent  = ReflectionAgent()
        result = agent.run(
            rag_result    = {"answer": "...", "sources": [...]},
            triage_result = {"urgency": "see_doctor_immediately", "reasoning": "..."},
            user_query    = "My knee is very painful…"
        )
        # result → {"approved": True/False, "revised_answer": "...", "notes": "..."}
    """

    def __init__(self):
        if not GROQ_API_KEY:
            raise ValueError(
                "GROQ_API_KEY is not set. "
                "Add it to your .env file and restart."
            )
        self.client = Groq(api_key=GROQ_API_KEY)
        self.model  = GROQ_MODEL

    def run(self, rag_result: dict, triage_result: dict, user_query: str) -> dict:
        """
        Reviews the RAGAgent's answer for safety and consistency.

        Parameters
        ----------
        rag_result    : Output from RAGAgent (must contain "answer").
        triage_result : Output from TriageAgent (must contain "urgency").
        user_query    : The user's original question (for context).

        Returns
        -------
        dict with keys:
            "approved"       : bool (was the original answer approved?)
            "revised_answer" : str  (the final answer to show the user)
            "notes"          : str  (what the reviewer changed/noted)
        """
        urgency      = triage_result.get("urgency", "unknown")
        original_ans = rag_result.get("answer", "")

        print(f"\n[ReflectionAgent] Reviewing answer (urgency: {urgency})...")
        print(f"[ReflectionAgent] Answer length: {len(original_ans)} chars.")

        # ── Build the review prompt ───────────────────────────────────────────
        user_message = (
            f"Original user query: {user_query}\n\n"
            f"Triage urgency: {urgency}\n"
            f"Triage reasoning: {triage_result.get('reasoning', '')}\n\n"
            f"AI-generated answer to review:\n{original_ans}\n\n"
            f"Please review this answer for safety and medical responsibility."
        )

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_message},
        ]

        # ── Call the Groq API ─────────────────────────────────────────────────
        try:
            response = self.client.chat.completions.create(
                model       = self.model,
                messages    = messages,
                temperature = 0.1,
                max_tokens  = 1200,
            )
            raw_output = response.choices[0].message.content.strip()
            print(f"[ReflectionAgent] Raw output (first 200 chars): {raw_output[:200]}...")

        except Exception as e:
            print(f"[ReflectionAgent] [Warning] API error: {e}")
            # If reflection fails, pass through the original answer unchanged
            return {
                "approved":       True,
                "revised_answer": original_ans,
                "notes":          f"Reflection skipped due to API error: {e}",
            }

        # ── Parse the JSON response ───────────────────────────────────────────
        return self._parse_output(raw_output, original_ans)

    def _parse_output(self, raw_output: str, fallback_answer: str) -> dict:
        """Safely parses the reviewer's JSON response."""
        try:
            result = json.loads(raw_output)

            # Ensure all required fields are present
            if "approved"       not in result: result["approved"]       = True
            if "revised_answer" not in result: result["revised_answer"] = fallback_answer
            if "notes"          not in result: result["notes"]          = ""

            return result

        except (json.JSONDecodeError, ValueError) as e:
            print(f"[ReflectionAgent] [Warning] JSON parse error: {e}")
            return {
                "approved":       True,
                "revised_answer": fallback_answer,
                "notes":          "JSON parse error — original answer passed through.",
            }


# ─────────────────────────────────────────────────────────────────────────────
# Quick test (run: python agents/reflection_agent.py)
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    agent = ReflectionAgent()

    # Test 1: Answer that needs fixing (urgent case, but no doctor referral)
    result = agent.run(
        rag_result    = {
            "answer": "Apply ice and rest your knee. You'll be fine in a few days.",
            "sources": []
        },
        triage_result = {
            "urgency":   "see_doctor_immediately",
            "reasoning": "Possible fracture — severe pain and swelling.",
        },
        user_query    = "I fell hard on my knee and it's very swollen and painful.",
    )
    print(f"\nTest 1 — Approved: {result['approved']}")
    print(f"Revised answer:\n{result['revised_answer']}")
    print(f"Notes: {result['notes']}")
