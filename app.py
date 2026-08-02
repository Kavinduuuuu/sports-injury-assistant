"""
app.py
──────
Sports Injury Recovery Assistant — Streamlit UI
================================================
This is the entry point for the web application.

Run with:
    streamlit run app.py

The UI:
  • Provides a chat-style input for injury descriptions
  • Shows a coloured urgency badge (triage result)
  • Displays the synthesised answer with cited sources
  • Shows retrieved knowledge-base excerpts in an expander
  • Shows per-agent timing and pipeline debug info in an expander
  • Always displays a medical disclaimer
"""

import streamlit as st
import json
import time
from pathlib import Path
from dotenv import load_dotenv

# Load .env so that API keys are available
load_dotenv()

# ─────────────────────────────────────────────────────────────────────────────
# Page configuration (must be the FIRST Streamlit call)
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title = "Sports Injury Recovery Assistant",
    page_icon  = "🏥",
    layout     = "wide",
    initial_sidebar_state = "collapsed",
)

# ─────────────────────────────────────────────────────────────────────────────
# Custom CSS — clean, professional look
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Google Font ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* ── Main background ── */
.main .block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 860px;
}

/* ── Urgency badges ── */
.badge-urgent {
    display: inline-block;
    background: linear-gradient(135deg, #ff4b4b, #c0392b);
    color: white;
    padding: 0.4rem 1.1rem;
    border-radius: 999px;
    font-weight: 600;
    font-size: 0.85rem;
    letter-spacing: 0.03em;
    margin-bottom: 0.5rem;
}
.badge-safe {
    display: inline-block;
    background: linear-gradient(135deg, #21ba45, #1a9e39);
    color: white;
    padding: 0.4rem 1.1rem;
    border-radius: 999px;
    font-weight: 600;
    font-size: 0.85rem;
    letter-spacing: 0.03em;
    margin-bottom: 0.5rem;
}

/* ── Answer card ── */
.answer-card {
    background: #f8f9fa;
    border-left: 4px solid #4a90e2;
    border-radius: 0 8px 8px 0;
    padding: 1.2rem 1.4rem;
    margin: 0.8rem 0;
    line-height: 1.7;
}

/* ── Source pill ── */
.source-pill {
    display: inline-block;
    background: #e8f0fe;
    color: #1a73e8;
    border-radius: 4px;
    padding: 0.15rem 0.5rem;
    font-size: 0.78rem;
    font-family: monospace;
    margin: 0.1rem;
}

/* ── Disclaimer box ── */
.disclaimer {
    background: #fff3cd;
    border: 1px solid #ffc107;
    border-radius: 8px;
    padding: 0.9rem 1.2rem;
    font-size: 0.85rem;
    color: #856404;
    margin-top: 1.5rem;
}

/* ── Timing chip ── */
.timing-chip {
    display: inline-block;
    background: #f1f3f5;
    border-radius: 4px;
    padding: 0.2rem 0.6rem;
    font-size: 0.78rem;
    color: #666;
    margin: 0.15rem;
    font-family: monospace;
}

/* ── Sidebar header ── */
.sidebar-header {
    font-size: 0.85rem;
    color: #888;
    margin-bottom: 0.3rem;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Lazy-load the Orchestrator (cached so it's only created once per session)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="⚙️  Loading AI agents…")
def load_orchestrator():
    """
    Instantiates the Orchestrator (and therefore all three agents) once.
    st.cache_resource keeps the same object alive across Streamlit reruns.
    Called ONLY when the user submits a query — not at startup.
    """
    from orchestrator import Orchestrator
    return Orchestrator()


@st.cache_resource(show_spinner=False)
def load_kb_stats():
    """
    Gets basic stats about the ChromaDB knowledge base for the sidebar.
    Uses ChromaDB directly (no embedding model) to avoid slow startup.
    """
    try:
        import chromadb
        import os
        chroma_path = os.getenv("CHROMA_DB_PATH", "./chroma_db")
        collection_name = os.getenv("CHROMA_COLLECTION", "sports_injury_kb")
        if not os.path.exists(chroma_path):
            return {"chunk_count": 0, "collection": collection_name,
                    "error": "Knowledge base not built yet. Run: python rag/ingest.py"}
        client = chromadb.PersistentClient(path=chroma_path)
        col = client.get_or_create_collection(collection_name)
        return {"chunk_count": col.count(), "collection": collection_name}
    except Exception as e:
        return {"chunk_count": 0, "collection": "unknown", "error": str(e)}


# ─────────────────────────────────────────────────────────────────────────────
# Main page
# ─────────────────────────────────────────────────────────────────────────────
st.title("🏋️ Sports Injury Recovery Assistant")
st.markdown(
    "Describe your sports injury or symptom below and get a grounded, "
    "AI-generated recovery guide backed by reference documents."
)

# Medical disclaimer — always visible
st.markdown("""
<div class="disclaimer">
⚕️ <strong>Medical Disclaimer:</strong> This tool provides general sports medicine 
information for educational purposes only. It is <strong>not a substitute for 
professional medical advice, diagnosis, or treatment</strong>. For serious injuries, 
<strong>consult a qualified doctor or physiotherapist immediately</strong>.
</div>
""", unsafe_allow_html=True)

# Inline Knowledge Base status warning (only shown if empty or error)
kb_stats = load_kb_stats()
chunk_count = kb_stats.get("chunk_count", 0)
if chunk_count == 0:
    st.error("⚠️ **Knowledge base is empty.** Please run `python rag/ingest.py` to index reference documents before using the assistant.")
    if kb_stats.get("error"):
        st.info(f"Database Error details: {kb_stats['error']}")

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# Query input
# ─────────────────────────────────────────────────────────────────────────────
if "query_text" not in st.session_state:
    st.session_state["query_text"] = ""

def update_query():
    st.session_state["query_text"] = st.session_state["temp_query"]

user_query = st.text_area(
    label       = "Describe your injury or symptom:",
    value       = st.session_state["query_text"],
    placeholder = (
        "e.g. I rolled my ankle during a football match. "
        "It's swollen and painful to walk on. What should I do?"
    ),
    height      = 100,
    key         = "temp_query",
    on_change   = update_query,
)

def clear_all():
    st.session_state.pop("last_result", None)
    st.session_state["query_text"] = ""
    st.session_state["temp_query"] = ""

col1, col2 = st.columns([1, 5])
with col1:
    submit = st.button("🔍 Analyse", type="primary", use_container_width=True)
with col2:
    st.button("🗑️ Clear", on_click=clear_all, use_container_width=False)



# ─────────────────────────────────────────────────────────────────────────────
# Run pipeline when user submits
# ─────────────────────────────────────────────────────────────────────────────
if submit:
    current_query = st.session_state["query_text"]
    if not current_query.strip():
        st.warning("Please describe your injury before submitting.")
    else:
        with st.spinner("Analyzing injury details..."):
            try:
                orchestrator = load_orchestrator()
                result = orchestrator.run(current_query.strip())
                if result.get("error"):
                    st.error(result["error"])
                else:
                    st.session_state["last_result"] = result
            except Exception as e:
                st.error(f"An unexpected error occurred: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# Helper to clean and format output
# ─────────────────────────────────────────────────────────────────────────────
def clean_display_answer(answer: str) -> str:
    if not answer:
        return ""
    trimmed = answer.strip()
    if trimmed.startswith("```"):
        lines = trimmed.split("\n")
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        trimmed = "\n".join(lines).strip()
        
    if (trimmed.startswith("{") and trimmed.endswith("}")) or (trimmed.startswith("[") and trimmed.endswith("]")):
        try:
            parsed = json.loads(trimmed, strict=False)
            if isinstance(parsed, dict):
                if "answer" in parsed:
                    return clean_display_answer(parsed["answer"])
                if "revised_answer" in parsed:
                    return clean_display_answer(parsed["revised_answer"])
                return "\n\n".join(f"{v}" for v in parsed.values() if isinstance(v, str))
            elif isinstance(parsed, list):
                return "\n\n".join(clean_display_answer(item) for item in parsed if isinstance(item, (str, dict)))
        except Exception:
            # Fallback for invalid JSON (e.g. unescaped double quotes inside values)
            import re
            key_match = re.search(r'"(?:answer|revised_answer)"\s*:\s*"', trimmed)
            if key_match:
                start_pos = key_match.end()
                end_pos = len(trimmed)
                boundary_match = re.search(r'"\s*,\s*"\w+"\s*:', trimmed[start_pos:])
                if boundary_match:
                    end_pos = start_pos + boundary_match.start()
                else:
                    boundary_match = re.search(r'"\s*}\s*$', trimmed)
                    if boundary_match:
                        end_pos = boundary_match.start()
                extracted = trimmed[start_pos:end_pos]
                extracted = extracted.replace('\\"', '"').replace('\\n', '\n').replace('\\t', '\t')
                return extracted.strip()
    return answer


# ─────────────────────────────────────────────────────────────────────────────
# Display results
# ─────────────────────────────────────────────────────────────────────────────
if "last_result" in st.session_state:
    result = st.session_state["last_result"]

    triage = result.get("triage", {})
    urgency = triage.get("urgency", "unknown")

    # ── Urgency Badge ─────────────────────────────────────────────────────────
    st.markdown("### 🚦 Urgency Assessment")

    if urgency == "see_doctor_immediately":
        badge_html = '<span class="badge-urgent">🚨 SEE A DOCTOR IMMEDIATELY</span>'
    elif urgency == "self_manageable":
        badge_html = '<span class="badge-safe">✅ SELF-MANAGEABLE</span>'
    else:
        badge_html = '<span class="badge-safe">⚠️ UNKNOWN</span>'

    st.markdown(badge_html, unsafe_allow_html=True)
    if triage.get("reasoning"):
        st.markdown(f"**Reasoning:** {triage['reasoning']}")

    # Urgent case warning
    if urgency == "see_doctor_immediately":
        st.error(
            "⚠️ **This injury may require immediate medical attention.** "
            "Please seek professional evaluation before attempting home treatment."
        )

    st.divider()

    # ── Main Answer ───────────────────────────────────────────────────────────
    st.markdown("### 💬 Recovery Guidance")

    raw_answer = result.get("final_answer", "")
    answer = clean_display_answer(raw_answer)
    st.markdown(answer)

    st.divider()

    # ── Bottom disclaimer ─────────────────────────────────────────────────────
    st.markdown("""
<div class="disclaimer">
⚕️ <strong>Reminder:</strong> This information is for 
educational purposes only. Always consult a qualified healthcare professional 
before making decisions about injury treatment, especially for serious injuries.
</div>
""", unsafe_allow_html=True)

