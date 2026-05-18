"""
Ingestion Module — PDF parsing, chunking, and summarization.
"""

import fitz  # PyMuPDF
from langchain_text_splitters import RecursiveCharacterTextSplitter
import ollama

from config import MODEL


def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract all text from a PDF file."""
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    doc.close()
    return text


def chunk_text(text: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> list[str]:
    """Split text into overlapping chunks for RAG.

    Args:
        text: The full document text
        chunk_size: Size of each chunk in characters
        chunk_overlap: Overlap between chunks in characters
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap, length_function=len,
    )
    return splitter.split_text(text)


def auto_chunk_size(doc_length: int) -> tuple[int, int]:
    """Auto-adjust chunk size based on document length.

    Returns:
        Tuple of (chunk_size, overlap)
    """
    if doc_length < 5000:
        return 500, 100
    elif doc_length < 20000:
        return 800, 150
    elif doc_length < 50000:
        return 1000, 200
    else:
        return 1500, 300


def summarize_chunks(chunks: list[str]) -> str:
    """Generate an executive summary using map-reduce over ALL chunks."""

    if len(chunks) <= 5:
        combined_text = "\n\n".join(chunks)
        return _generate_summary(combined_text)

    # Map phase: summarize groups of 5 chunks each
    mini_summaries = []
    for i in range(0, len(chunks), 5):
        group = chunks[i:i + 5]
        combined = "\n\n".join(group)
        prompt = f"""Summarize the following section of a document in 3-4 bullet points.
Capture key facts, names, dates, obligations, and numbers. Be specific, not vague.

--- TEXT ---
{combined}
--- END ---

Bullet point summary:"""
        response = ollama.chat(model=MODEL, messages=[{"role": "user", "content": prompt}])
        mini_summaries.append(response["message"]["content"])

    # Reduce phase: combine all mini-summaries into one executive summary
    all_summaries = "\n\n".join(
        f"[Section {i+1}]:\n{s}" for i, s in enumerate(mini_summaries)
    )
    return _generate_summary(all_summaries)


def _generate_summary(text: str) -> str:
    """Generate a structured executive summary from text."""
    prompt = f"""You are a document analysis assistant. Below is content from a document (or summaries of its sections).
Please provide a concise executive summary that includes:
1. What type of document this is
2. The key parties involved
3. Main obligations and terms
4. Important dates or deadlines
5. Any critical clauses or conditions

Keep the summary clear and structured so it can be used as a briefing.

--- CONTENT ---
{text}
--- END ---

Executive Summary:"""
    response = ollama.chat(model=MODEL, messages=[{"role": "user", "content": prompt}])
    return response["message"]["content"]
