"""
rag/ingest.py
─────────────
RAG Ingestion Pipeline
======================
Reads documents (PDFs and .txt files) from the /data folder,
splits them into chunks, embeds each chunk with a local sentence-transformer
model, and saves everything into a persistent ChromaDB collection.

Run once (or whenever you add new documents):
    python rag/ingest.py
"""

import os
import sys
import json
import hashlib
import textwrap
from pathlib import Path

# ── Load environment variables from .env ─────────────────────────────────────
from dotenv import load_dotenv
load_dotenv()

# ── Third-party libraries ─────────────────────────────────────────────────────
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
from pypdf import PdfReader

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION (read from .env with sensible defaults)
# ─────────────────────────────────────────────────────────────────────────────
CHROMA_DB_PATH    = os.getenv("CHROMA_DB_PATH", "./chroma_db")
CHROMA_COLLECTION = os.getenv("CHROMA_COLLECTION", "sports_injury_kb")
EMBEDDING_MODEL   = "all-MiniLM-L6-v2"   # small, fast, free, runs locally
DATA_FOLDER       = Path(__file__).parent.parent / "data"

# Chunking settings
CHUNK_SIZE_TOKENS    = 400   # approximate tokens per chunk
CHUNK_OVERLAP_TOKENS = 50    # overlap so chunks share context at boundaries
WORDS_PER_TOKEN      = 0.75  # rough conversion: tokens ≈ words / 0.75


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — Read raw text from a file (PDF or plain text)
# ─────────────────────────────────────────────────────────────────────────────
def load_document(file_path: Path) -> str:
    """
    Reads a file and returns its full text content as a single string.
    Supports .pdf and .txt (and .md) files.
    """
    suffix = file_path.suffix.lower()

    if suffix == ".pdf":
        reader = PdfReader(str(file_path))
        pages  = [page.extract_text() or "" for page in reader.pages]
        return "\n\n".join(pages)

    elif suffix in (".txt", ".md"):
        return file_path.read_text(encoding="utf-8", errors="replace")

    else:
        print(f"  [SKIP] Unsupported file type: {file_path.name}")
        return ""


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — Split text into overlapping chunks
# ─────────────────────────────────────────────────────────────────────────────
def chunk_text(text: str, source_name: str) -> list[dict]:
    """
    Splits a long text into smaller chunks with slight overlap.

    Strategy:
    - Split on double newlines first (paragraph breaks / headings).
    - Combine paragraphs until the chunk hits CHUNK_SIZE_TOKENS.
    - Keep the last CHUNK_OVERLAP_TOKENS words of the previous chunk at the
      start of the next one so context is not lost at boundaries.

    Returns a list of dicts:
        {"chunk_id": "...", "text": "...", "source": "...", "chunk_index": n}
    """
    # Convert token targets to approximate word counts
    max_words     = int(CHUNK_SIZE_TOKENS / WORDS_PER_TOKEN)   # ~533 words
    overlap_words = int(CHUNK_OVERLAP_TOKENS / WORDS_PER_TOKEN) # ~67 words

    # Split into paragraphs; filter out blank lines
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

    chunks      = []
    current     = []   # words accumulated for the current chunk
    chunk_index = 0

    for para in paragraphs:
        words = para.split()

        # If adding this paragraph would overflow the chunk, flush first
        if current and (len(current) + len(words)) > max_words:
            chunk_text_str = " ".join(current)
            chunk_id       = _make_chunk_id(source_name, chunk_index)
            chunks.append({
                "chunk_id":    chunk_id,
                "text":        chunk_text_str,
                "source":      source_name,
                "chunk_index": chunk_index,
            })
            chunk_index += 1

            # Keep the overlap tail for next chunk
            current = current[-overlap_words:] if overlap_words else []

        current.extend(words)

    # Don't forget the last partial chunk
    if current:
        chunk_text_str = " ".join(current)
        chunk_id       = _make_chunk_id(source_name, chunk_index)
        chunks.append({
            "chunk_id":    chunk_id,
            "text":        chunk_text_str,
            "source":      source_name,
            "chunk_index": chunk_index,
        })

    return chunks


def _make_chunk_id(source_name: str, index: int) -> str:
    """Creates a stable, unique ID for a chunk (used as ChromaDB document ID)."""
    raw = f"{source_name}::chunk_{index}"
    return hashlib.md5(raw.encode()).hexdigest()[:12] + f"_{index}"


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — Embed chunks and store in ChromaDB
# ─────────────────────────────────────────────────────────────────────────────
def ingest_all():
    """
    Main ingestion function:
    1. Scans /data for supported files.
    2. Reads and chunks each file.
    3. Embeds chunks with sentence-transformers.
    4. Upserts them into the ChromaDB collection (safe to re-run).
    """
    # ── Validate the data folder ──────────────────────────────────────────────
    if not DATA_FOLDER.exists():
        print(f"[ERROR] Data folder not found: {DATA_FOLDER}")
        print("  Create the folder and drop your PDF/TXT documents inside it.")
        sys.exit(1)

    supported_files = list(DATA_FOLDER.glob("*.pdf")) + \
                      list(DATA_FOLDER.glob("*.txt")) + \
                      list(DATA_FOLDER.glob("*.md"))

    if not supported_files:
        print(f"[WARNING] No PDF/TXT/MD files found in {DATA_FOLDER}")
        print("  Add some sports-injury reference documents and re-run.")
        return

    # ── Load embedding model (downloaded once, cached locally) ───────────────
    print(f"\n[1/4] Loading embedding model: {EMBEDDING_MODEL}")
    embedder = SentenceTransformer(EMBEDDING_MODEL)

    # ── Connect to (or create) the ChromaDB collection ───────────────────────
    print(f"[2/4] Connecting to ChromaDB at: {CHROMA_DB_PATH}")
    client     = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    collection = client.get_or_create_collection(
        name     = CHROMA_COLLECTION,
        metadata = {"hnsw:space": "cosine"},  # cosine similarity for semantic search
    )

    # ── Process each file ─────────────────────────────────────────────────────
    print(f"[3/4] Processing {len(supported_files)} file(s)...\n")
    all_chunks = []

    for file_path in supported_files:
        print(f"  -> {file_path.name}")
        raw_text = load_document(file_path)
        if not raw_text.strip():
            print(f"     [SKIP] Empty or unreadable file.")
            continue

        chunks = chunk_text(raw_text, file_path.name)
        print(f"     Created {len(chunks)} chunks.")
        all_chunks.extend(chunks)

    if not all_chunks:
        print("\n[WARNING] No chunks were created. Check your documents.")
        return

    # ── Embed and upsert in batches ───────────────────────────────────────────
    print(f"\n[4/4] Embedding {len(all_chunks)} chunks and storing in ChromaDB...")

    BATCH_SIZE = 64   # embed this many at once to avoid memory spikes
    for i in range(0, len(all_chunks), BATCH_SIZE):
        batch  = all_chunks[i : i + BATCH_SIZE]
        texts  = [c["text"]     for c in batch]
        ids    = [c["chunk_id"] for c in batch]
        metas  = [
            {"source": c["source"], "chunk_index": c["chunk_index"]}
            for c in batch
        ]

        # Embed the raw text (returns a numpy array; ChromaDB accepts lists)
        embeddings = embedder.encode(texts, show_progress_bar=False).tolist()

        # upsert = insert OR update if the ID already exists (safe to re-run)
        collection.upsert(
            ids        = ids,
            documents  = texts,
            embeddings = embeddings,
            metadatas  = metas,
        )
        print(f"  Stored batch {i // BATCH_SIZE + 1} "
              f"({min(i + BATCH_SIZE, len(all_chunks))}/{len(all_chunks)} chunks)")

    print(f"\n[Success] Ingestion complete! "
          f"Collection '{CHROMA_COLLECTION}' now has "
          f"{collection.count()} document chunks.")
    print(f"   DB location: {Path(CHROMA_DB_PATH).resolve()}\n")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    ingest_all()
