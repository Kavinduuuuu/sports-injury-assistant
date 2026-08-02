"""
agents/triage_agent.py
──────────────────────
TriageAgent — Router Pattern
============================
The first agent in the pipeline. It reads the user's symptom description
and quickly classifies how urgent the situation is:

  • "self_manageable"        → the injury sounds minor; home care is appropriate
  • "see_doctor_immediately" → symptoms suggest something serious; see a doctor

Design pattern: Router — a lightweight model makes a fast binary decision
that routes the rest of the pipeline's behavior.

Model: Groq + Llama 3.1 8B Instant (very fast, cheap, good for classification)
"""

import os
import json
import re
from dotenv import load_dotenv

load_dotenv()

from groq import Groq

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL   = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")

# The system prompt tells the model exactly what JSON format to produce.
# Being explicit about the output format is crucial for reliable parsing.
SYSTEM_PROMPT = """You are a sports injury triage assistant.
Your job is to classify the urgency of an injury description.

Rules:
1. Classify as "see_doctor_immediately" if the description mentions ANY of:
   - Loss of consciousness, dizziness, or confusion
   - Severe swelling, numbness, or tingling
   - Visible deformity or suspected fracture/dislocation
   - Inability to bear weight or use a limb at all
   - Head, neck, or spinal injury
   - Chest pain or difficulty breathing
   - Severe bleeding that won't stop
2. Otherwise classify as "self_manageable".
3. Keep your reasoning brief (1-2 sentences).

IMPORTANT: Respond ONLY with valid JSON in this exact format:
{
  "urgency": "self_manageable" or "see_doctor_immediately",
  "reasoning": "brief explanation"
}
Do NOT include any text before or after the JSON object."""


# ─────────────────────────────────────────────────────────────────────────────
# Agent class
# ─────────────────────────────────────────────────────────────────────────────
class TriageAgent:
    """
    Routes an injury query to one of two urgency levels.

    Usage:
        agent  = TriageAgent()
        result = agent.run("My ankle is swollen and painful after a fall.")
        # result → {"urgency": "self_manageable", "reasoning": "..."}
    """

    def __init__(self):
        if not GROQ_API_KEY:
            raise ValueError(
                "GROQ_API_KEY is not set. "
                "Add it to your .env file and restart."
            )
        self.client = Groq(api_key=GROQ_API_KEY)
        self.model  = GROQ_MODEL

    def run(self, user_query: str) -> dict:
        """
        Classifies the urgency of a sports injury description.

        Parameters
        ----------
        user_query : The user's free-text symptom description.

        Returns
        -------
        dict with keys:
            "urgency"   : "self_manageable" | "see_doctor_immediately"
            "reasoning" : str (brief explanation)

        On error, returns a safe default so the pipeline keeps running.
        """
        print(f"\n[TriageAgent] Input: {user_query[:80]}...")

        # ── Build the message for the LLM ────────────────────────────────────
        messages = [
            {"role": "system",  "content": SYSTEM_PROMPT},
            {"role": "user",    "content": user_query},
        ]

        # ── Call the Groq API ─────────────────────────────────────────────────
        try:
            response = self.client.chat.completions.create(
                model       = self.model,
                messages    = messages,
                temperature = 0.1,   # low temperature → more deterministic output
                max_tokens  = 200,   # triage answer is always short
            )
            raw_output = response.choices[0].message.content.strip()
            print(f"[TriageAgent] Raw output: {raw_output}")

        except Exception as e:
            # If the API call fails, default to "see_doctor_immediately"
            # so we never accidentally downplay a potentially serious injury.
            print(f"[TriageAgent] [Warning] API error: {e}")
            return {
                "urgency":   "see_doctor_immediately",
                "reasoning": f"Could not complete triage due to API error: {e}. Defaulting to caution.",
            }

        # ── Parse the JSON response ───────────────────────────────────────────
        return self._parse_output(raw_output)

    def _parse_output(self, raw_output: str) -> dict:
        """
        Safely parses the LLM's JSON response.
        Falls back to a safe default if parsing fails.
        """
        try:
            result = json.loads(raw_output)

            # Validate the expected fields exist
            if "urgency" not in result or "reasoning" not in result:
                raise ValueError("Missing required fields in JSON response.")

            # Normalise the urgency value (just in case the model adds typos)
            if result["urgency"] not in ("self_manageable", "see_doctor_immediately"):
                result["urgency"] = "see_doctor_immediately"

            return result

        except (json.JSONDecodeError, ValueError) as e:
            print(f"[TriageAgent] [Warning] JSON parse error: {e}")
            # Try to extract urgency with a regex as a last resort
            if "self_manageable" in raw_output:
                urgency = "self_manageable"
            else:
                urgency = "see_doctor_immediately"   # default to caution

            return {
                "urgency":   urgency,
                "reasoning": "Parsed from unstructured response (JSON error).",
            }


# ─────────────────────────────────────────────────────────────────────────────
# Quick test (run: python agents/triage_agent.py)
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    agent = TriageAgent()

    test_cases = [
        "I rolled my ankle during a football match. It's a bit swollen but I can walk.",
        "I fell on my wrist doing gymnastics and now it's severely deformed and I can't move it.",
        "My shoulder is sore after a heavy gym session.",
    ]

    for query in test_cases:
        print("\n" + "-" * 60)
        result = agent.run(query)
        print(f"Result: {json.dumps(result, indent=2)}")
