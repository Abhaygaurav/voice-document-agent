"""
DocTalk — Voice Document Agent
Upload a PDF → Get an AI summary → Ask questions by voice or text.
Fully local: Ollama + ChromaDB + Browser APIs. No API keys needed.

Run: python app.py
Open: http://localhost:8000 (Chrome recommended for voice)
"""

import os
import tempfile
from pathlib import Path

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse

from config import MODEL, EMBED_MODEL
from ingestion import extract_text_from_pdf, chunk_text, auto_chunk_size, summarize_chunks
from rag import store_chunks, chat_with_document, session, reset_session

app = FastAPI(title="DocTalk — Voice Document Agent")

TEMPLATE_PATH = Path(__file__).parent / "templates" / "index.html"


# ─── API ROUTES ──────────────────────────────────────────────────────────────────

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    """Upload PDF → extract → chunk → embed → summarize."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        # Extract text
        raw_text = extract_text_from_pdf(tmp_path)

        # Auto-adjust chunk size based on document length
        chunk_size, overlap = auto_chunk_size(len(raw_text))
        chunks = chunk_text(raw_text, chunk_size=chunk_size, chunk_overlap=overlap)

        # Generate summary (map-reduce for full coverage)
        summary = summarize_chunks(chunks)

        # Store chunks in vector DB for RAG
        store_chunks(chunks)

        # Update session
        session["summary"] = summary
        session["chunks"] = chunks
        session["conversation"] = []

        return {
            "source_file": file.filename,
            "total_characters": len(raw_text),
            "total_chunks": len(chunks),
            "summary": summary,
        }
    finally:
        os.unlink(tmp_path)


@app.post("/chat")
async def chat_endpoint(data: dict):
    """Chat with the document using RAG."""
    user_message = data.get("message", "")
    if user_message == "__clear__":
        reset_session()
        return {"reply": "Conversation cleared."}
    if not session["summary"]:
        return {"reply": "Please upload a document first."}
    return {"reply": chat_with_document(user_message)}


@app.get("/", response_class=HTMLResponse)
async def home():
    """Serve the frontend."""
    return TEMPLATE_PATH.read_text()


# ─── ENTRY POINT ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    print("\n✨ DocTalk — Voice Document Agent")
    print("📍 Open http://localhost:8000 in Chrome")
    print("   Fully local — no API keys needed\n")
    uvicorn.run(app, host="0.0.0.0", port=8000)
