"""
run_ingest.py
─────────────
Helper script to run the RAG ingestion pipeline.
Use this when the direct `python rag/ingest.py` call doesn't work
(e.g. Windows Store Python quirks).

Run with the same Python that has the packages installed:
    streamlit run run_ingest.py
  OR simply double-click in Windows Explorer.
"""
import subprocess, sys, os

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rag.ingest import ingest_all
ingest_all()
