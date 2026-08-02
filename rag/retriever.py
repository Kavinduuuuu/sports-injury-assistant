"""
rag/retriever.py
────────────────
RAG Retriever
=============
Loads the ChromaDB collection that was built by rag/ingest.py and
exposes a single function: retrieve(query, top_k) → list of chunk dicts.

The RAGAgent calls this as a "tool" — it passes a natural-language query
and gets back the most relevant knowledge-base chunks.
"""

import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
load_dotenv()

from sentence_transformers import SentenceTransformer
import chromadb

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────
CHROMA_DB_PATH    = os.getenv("CHROMA_DB_PATH", "./chroma_db")
CHROMA_COLLECTION = os.getenv("CHROMA_COLLECTION", "sports_injury_kb")
EMBEDDING_MODEL   = "all-MiniLM-L6-v2"
DEFAULT_TOP_K     = int(os.getenv("RAG_TOP_K", "4"))


# ─────────────────────────────────────────────────────────────────────────────
# Module-level singletons (loaded once, reused across calls)
# ─────────────────────────────────────────────────────────────────────────────
# We use Optional so type checkers are happy; they get set in _ensure_loaded().
_embedder:   Optional[SentenceTransformer] = None
_collection: Optional[chromadb.Collection]  = None


def _ensure_loaded():
    """
    Lazy-loads the embedding model and ChromaDB collection.
    Called automatically the first time retrieve() is used.
    This avoids slow startup when other parts of the app don't need retrieval.
    """
    global _embedder, _collection

    if _embedder is None:
        print("[Retriever] Loading embedding model...")
        _embedder = SentenceTransformer(EMBEDDING_MODEL)

    if _collection is None:
        print(f"[Retriever] Connecting to ChromaDB at '{CHROMA_DB_PATH}'...")
        client     = chromadb.PersistentClient(path=CHROMA_DB_PATH)
        _collection = client.get_or_create_collection(
            name     = CHROMA_COLLECTION,
            metadata = {"hnsw:space": "cosine"},
        )
        count = _collection.count()
        print(f"[Retriever] Collection '{CHROMA_COLLECTION}' loaded - {count} chunks.")
        if count == 0:
            print("[Retriever] [Warning] Collection is empty! Run: python rag/ingest.py")


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────
def retrieve(query: str, top_k: int = DEFAULT_TOP_K) -> list[dict]:
    """
    Searches the knowledge base for chunks relevant to `query`.

    Parameters
    ----------
    query  : The user's symptom description or a reformulated search query.
    top_k  : How many chunks to return (default set in .env as RAG_TOP_K).

    Returns
    -------
    A list of dicts, each with:
        {
            "chunk_id":    str,   # unique ID (used for citation)
            "text":        str,   # the chunk's text content
            "source":      str,   # original filename
            "chunk_index": int,   # position within source file
            "distance":    float  # lower = more similar (cosine distance)
        }
    Returns an empty list if the collection is empty or an error occurs.
    """
    _ensure_loaded()

    if _collection.count() == 0:
        return []  # nothing to retrieve — ingest.py hasn't been run yet

    # Embed the query with the same model used during ingestion
    query_embedding = _embedder.encode([query]).tolist()

    # Query ChromaDB for the top-k nearest chunks
    results = _collection.query(
        query_embeddings = query_embedding,
        n_results        = min(top_k, _collection.count()),  # can't exceed total
        include          = ["documents", "metadatas", "distances"],
    )

    # Reshape ChromaDB's nested-list output into a clean list of dicts
    chunks = []
    for i, doc in enumerate(results["documents"][0]):
        meta = results["metadatas"][0][i]
        chunks.append({
            "chunk_id":    results["ids"][0][i],
            "text":        doc,
            "source":      meta.get("source", "unknown"),
            "chunk_index": meta.get("chunk_index", i),
            "distance":    results["distances"][0][i],
        })

    return chunks


def get_collection_stats() -> dict:
    """
    Returns basic stats about the knowledge base.
    Useful for the Streamlit UI to show users if the KB is loaded.
    """
    _ensure_loaded()
    return {
        "chunk_count": _collection.count(),
        "collection":  CHROMA_COLLECTION,
        "db_path":     CHROMA_DB_PATH,
    }
