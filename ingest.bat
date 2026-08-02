@echo off
REM ─────────────────────────────────────────────────────────────────────────
REM ingest.bat  —  Run the RAG ingestion pipeline
REM
REM Double-click this file, or run from Command Prompt:
REM    ingest.bat
REM
REM This will:
REM   1. Read all PDF/TXT files from the /data folder
REM   2. Chunk and embed them with sentence-transformers
REM   3. Store embeddings in ChromaDB (./chroma_db/)
REM ─────────────────────────────────────────────────────────────────────────
cd /d "%~dp0"
echo Running RAG ingestion pipeline...
echo.

REM Try to find the right Python (Windows Store Python or system Python)
SET PYEXE=python

IF EXIST "%LOCALAPPDATA%\Microsoft\WindowsApps\python3.12.exe" (
    SET PYEXE="%LOCALAPPDATA%\Microsoft\WindowsApps\python3.12.exe"
)

%PYEXE% rag\ingest.py
IF %ERRORLEVEL% NEQ 0 (
    echo.
    echo ERROR: Ingestion failed. Make sure you have run:
    echo    pip install -r requirements.txt
    echo.
)
pause
