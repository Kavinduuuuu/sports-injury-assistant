# 🏋️ Sports Injury Recovery Assistant

> **University Assignment — Agentic RAG System demonstrating multi-agent orchestration, agent-to-agent communication, and Retrieval-Augmented Generation (RAG) for sports medicine first aid.**

---

## 📋 Project Description

The Sports Injury Recovery Assistant is a Python-based agentic AI web application built with Streamlit. A user describes a sports injury or symptom in natural language; the system routes the query through a three-agent pipeline that triages urgency, retrieves grounded information from a local knowledge base, reflects on the answer for medical safety, and returns a cited, responsible recovery guide.

**Key concepts demonstrated:**
- Agentic AI design patterns (Router, Tool-use/RAG, Reflection)
- Agent-to-agent communication via structured JSON messages
- Multi-model orchestration (Groq + OpenRouter)
- Retrieval-Augmented Generation (RAG) with ChromaDB and sentence-transformers
- Responsible AI: automatic safety review before responses reach the user

---

## 🏗️ Architecture Diagram

```
User (Streamlit UI)
        │
        ▼
  ┌─────────────┐     Structured JSON message
  │  Triage     │ ──────────────────────────────────────────────►
  │  Agent      │  { "urgency": "...", "reasoning": "..." }
  │ (Groq LLM)  │
  └─────────────┘
        │
        ▼
  ┌─────────────┐  Tool call: retrieve(query)    ┌──────────────┐
  │   RAG       │ ──────────────────────────────► │  ChromaDB    │
  │   Agent     │ ◄────────────────── top-k chunks│  (local)     │
  │(OpenRouter) │                                 └──────────────┘
  └─────────────┘
        │ Structured JSON message
        │  { "answer": "...", "sources": [...] }
        ▼
  ┌─────────────┐
  │ Reflection  │
  │   Agent     │
  │ (Groq LLM)  │
  └─────────────┘
        │
        ▼  { "approved": bool, "revised_answer": "...", "notes": "..." }
        │
        ▼
  Streamlit UI (shows urgency badge + answer + cited sources)
```

---

## ⚙️ Setup Instructions

### 1. Clone the repository
```bash
git clone <your-repo-url>
cd sports-injury-assistant
```

### 2. Create and activate a virtual environment
```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure API keys
```bash
cp .env.example .env
# Open .env and fill in your real API keys:
#   GROQ_API_KEY       → https://console.groq.com/keys
#   OPENROUTER_API_KEY → https://openrouter.ai/keys
```

### 5. Add documents to /data and run ingestion
Place any `.pdf` or `.txt` sports injury reference documents in the `/data` folder.  
Two example documents are already included. Then run:
```bash
python rag/ingest.py
```
This downloads the embedding model (once), chunks your documents, and stores them in ChromaDB.

### 6. Run the Streamlit app
```bash
streamlit run app.py
```
Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## 🤖 Agent Summary & Model-Choice Comparison

| Agent | Model | Provider | Why This Model? | Pattern |
|---|---|---|---|---|
| **TriageAgent** | `llama-3.1-8b-instant` | Groq | Ultra-fast inference (<1s). Simple binary classification needs speed, not depth. Free tier available. | Router |
| **RAGAgent** | `anthropic/claude-3-haiku` | OpenRouter | Higher reasoning. Needs to synthesize multiple retrieved chunks into a coherent, cited answer. More capable at following complex output instructions. | Tool-use / RAG |
| **ReflectionAgent** | `llama-3.1-8b-instant` | Groq | Fast, cheap. Safety review needs common sense, not deep reasoning. Low latency keeps total pipeline time acceptable. | Reflection |

**Groq** is used for the "fast" agents because its LPU hardware gives sub-second latency at very low cost — ideal for classification and review tasks.  
**OpenRouter** is used for the RAGAgent because it provides access to frontier models (Claude, GPT-4-class) through a single OpenAI-compatible API, without managing multiple provider accounts.

---

## 💬 Agent Communication Sequence Diagram

```
User          Orchestrator    TriageAgent     RAGAgent     ReflectionAgent   ChromaDB
 │                 │               │              │               │              │
 │── query ───────►│               │              │               │              │
 │                 │── run(query) ─►│              │               │              │
 │                 │               │              │               │              │
 │                 │◄── JSON ──────│              │               │              │
 │                 │  {urgency,    │              │               │              │
 │                 │   reasoning}  │              │               │              │
 │                 │               │              │               │              │
 │                 │── run(query, triage) ────────►│              │              │
 │                 │               │              │── retrieve() ─┼─────────────►│
 │                 │               │              │◄─ top-k chunks┼──────────────│
 │                 │               │              │  (tool result)│              │
 │                 │               │              │               │              │
 │                 │◄──────────────┼── JSON ──────│               │              │
 │                 │               │  {answer,    │               │              │
 │                 │               │   sources}   │               │              │
 │                 │               │              │               │              │
 │                 │── run(rag_result, triage) ───────────────────►│             │
 │                 │               │              │               │              │
 │                 │◄──────────────┼──────────────┼── JSON ───────│              │
 │                 │               │              │  {approved,   │              │
 │                 │               │              │   revised_ans}│              │
 │                 │               │              │               │              │
 │◄── final ───────│               │              │               │              │
 │    result       │               │              │               │              │
```

---

## 📚 RAG Pipeline Explanation

```
/data/*.pdf, *.txt
        │
        ▼
   load_document()          ← Reads PDF pages or plain text
        │
        ▼
   chunk_text()             ← Splits into ~400-token chunks with ~50-token overlap
        │                      (paragraph-boundary aware)
        ▼
   SentenceTransformer      ← all-MiniLM-L6-v2 (runs locally, no API needed)
   .encode(chunks)          ← Produces 384-dimensional dense embedding vectors
        │
        ▼
   ChromaDB.upsert()        ← Stores text + embeddings in a persistent local collection
        │                      (safe to re-run; upsert avoids duplicates)
        ▼
   chroma_db/               ← Persistent storage on disk

── At query time ──────────────────────────────────────────────────────────

   user_query
        │
        ▼
   SentenceTransformer      ← Same model embeds the query
   .encode(query)
        │
        ▼
   ChromaDB.query()         ← Cosine similarity search → top-k nearest chunks
        │
        ▼
   Retrieved chunks          ← Passed to RAGAgent as context (cited in answer)
```

**Why sentence-transformers?**  
The `all-MiniLM-L6-v2` model runs entirely locally with no API key, is fast, and produces good semantic embeddings for English text. It's ideal for a project that needs to work offline or at zero cost.

**Why ChromaDB?**  
ChromaDB is easy to set up, requires no external server, and persists data to disk automatically. It supports cosine similarity search natively.

---

## 🔗 Live Demo Link

<!-- TODO: Add your Streamlit Cloud / Hugging Face Spaces deployment link here -->
_Demo link: coming soon_

---

## ⚠️ Known Limitations

1. **Knowledge base depth**: The system can only answer questions about topics covered in the ingested documents. Questions outside the knowledge base will receive a "not found" response.

2. **No memory / conversation history**: Each query is processed independently. The agents do not remember previous turns in the session.

3. **Triage is probabilistic**: The TriageAgent uses an LLM for classification, which means edge cases may be misclassified. The system always defaults to "see doctor" on API errors as a safety measure.

4. **Not a medical device**: This application is for educational demonstration only. It has not been validated clinically and must not be used for real medical decision-making.

5. **API rate limits**: Both Groq and OpenRouter have rate limits on free tiers. Under heavy usage, requests may be throttled.

6. **RAG hallucination risk**: While the RAGAgent is instructed to only use retrieved chunks, LLMs can still generate content not present in the sources. The ReflectionAgent adds a safety layer but cannot guarantee complete accuracy.

7. **Embedding model limitations**: `all-MiniLM-L6-v2` is optimised for general English text. Highly technical medical terminology may not retrieve as precisely as a domain-specific biomedical embedding model.

---

## 📁 Project Structure

```
sports-injury-assistant/
├── agents/
│   ├── __init__.py
│   ├── triage_agent.py       # TriageAgent (Groq) — Router pattern
│   ├── rag_agent.py          # RAGAgent (OpenRouter) — Tool-use/RAG pattern
│   └── reflection_agent.py   # ReflectionAgent (Groq) — Reflection pattern
├── rag/
│   ├── __init__.py
│   ├── ingest.py             # Reads /data docs → chunks → ChromaDB
│   └── retriever.py          # retrieve(query) tool called by RAGAgent
├── data/
│   ├── sports_injury_guide.txt
│   └── physiotherapy_protocols.txt
├── chroma_db/                # Auto-created by ingest.py (git-ignored)
├── orchestrator.py           # Sequential agent pipeline + message passing
├── app.py                    # Streamlit web UI
├── .env                      # Your real API keys (git-ignored)
├── .env.example              # Template — copy this to .env
├── .gitignore
├── requirements.txt
└── README.md
```

# 🏥 Sports Injury Recovery Assistant
An agentic RAG-based AI assistant that provides grounded, evidence-based first-aid and recovery 
guidance for common sports and karate injuries. Built for IT41043 (Agentic AI).
## 🌐 Live Demo
**[Try it here](YOUR_STREAMLIT_URL_HERE)**
## 📋 Project Description
This application addresses a real problem faced by athletes and martial artists (including karate 
practitioners): getting quick, reliable first-aid guidance for common training injuries before 
deciding whether professional medical care is needed. It uses a 3-agent pipeline combined with a 
Retrieval-Augmented Generation (RAG) system grounded in a curated knowledge base of 33 sports 
medicine documents.
**⚠️ Disclaimer:** This tool is for educational purposes only and is not a substitute for 
professional medical advice.
## ⚙️ Setup Instructions
1. Clone the repository:
   \`\`\`bash
   git clone https://github.com/YOUR_USERNAME/sports-injury-assistant.git
   cd sports-injury-assistant
   \`\`\`
2. Install dependencies:
   \`\`\`bash
   pip install -r requirements.txt
   \`\`\`
3. Set up environment variables:
   \`\`\`bash
   cp .env.example .env
   # Fill in GROQ_API_KEY and OPENROUTER_API_KEY
   \`\`\`
4. Run the RAG ingestion pipeline (embeds knowledge base into ChromaDB):
   \`\`\`bash
   python rag/ingest.py
   \`\`\`
5. Launch the app:
   \`\`\`bash
   streamlit run app.py
   \`\`\`

## 🔄 Agent Communication Sequence

\`\`\`mermaid
sequenceDiagram
    participant U as User
    participant O as Orchestrator
    participant T as TriageAgent
    participant R as RAGAgent
    participant Rf as ReflectionAgent
    participant DB as ChromaDB

    U->>O: Injury description
    O->>T: {"query": "..."}
    T->>O: {"urgency": "self_manageable/see_doctor", "reasoning": "..."}
    O->>R: {"query": "...", "urgency_context": "..."}
    R->>DB: Retrieve top-k chunks
    DB->>R: Relevant excerpts
    R->>O: {"answer": "...", "sources": [...]}
    O->>Rf: {"answer": "...", "urgency": "..."}
    Rf->>O: {"approved": true, "revised_answer": "..."}
    O->>U: Final response
\`\`\`