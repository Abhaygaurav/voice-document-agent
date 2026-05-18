"""
RAG Module — Vector store management and retrieval-augmented chat.
"""

import ollama
import chromadb

from config import MODEL, EMBED_MODEL


# ChromaDB client (in-memory)
chroma_client = chromadb.Client()
collection = None

# Session state (single-user)
session = {
    "summary": "",
    "chunks": [],
    "conversation": [],
}


def store_chunks(chunks: list[str]):
    """Embed and store all chunks in ChromaDB."""
    global collection

    try:
        chroma_client.delete_collection("document_chunks")
    except Exception:
        pass

    collection = chroma_client.create_collection(
        name="document_chunks", metadata={"hnsw:space": "cosine"},
    )

    batch_size = 20
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        ids = [f"chunk_{j}" for j in range(i, i + len(batch))]
        embeddings = []
        for chunk in batch:
            resp = ollama.embed(model=EMBED_MODEL, input=chunk)
            embeddings.append(resp["embeddings"][0])
        collection.add(ids=ids, documents=batch, embeddings=embeddings)


def retrieve_relevant_chunks(query: str, n_results: int = 4) -> list[str]:
    """Search ChromaDB for chunks most relevant to the query."""
    if collection is None:
        return []

    try:
        query_embedding = ollama.embed(model=EMBED_MODEL, input=query)
        results = collection.query(
            query_embeddings=[query_embedding["embeddings"][0]],
            n_results=n_results,
        )
        if results and results["documents"] and results["documents"][0]:
            return results["documents"][0]
    except Exception as e:
        print(f"RAG search error: {e}")

    return []


def chat_with_document(user_message: str) -> str:
    """Answer questions using RAG — retrieves relevant chunks from ChromaDB."""
    system_prompt = f"""You are a helpful voice assistant that has analyzed a document.
You speak naturally and concisely — keep responses to 2-3 sentences since this is voice.
Use the document summary AND the relevant excerpts below to answer questions.
If the answer is in the excerpts, quote or reference the specific text.
If you cannot find the answer in the provided text, say so honestly.

--- DOCUMENT SUMMARY ---
{session['summary']}
--- END SUMMARY ---"""

    # Retrieve relevant chunks
    retrieved_chunks = retrieve_relevant_chunks(user_message)
    context_text = ""
    if retrieved_chunks:
        context_text = "\n\n".join(
            f"[Excerpt {i+1}]: {chunk}" for i, chunk in enumerate(retrieved_chunks)
        )

    # Build augmented prompt
    if context_text:
        augmented_message = f"""Question: {user_message}

--- RELEVANT DOCUMENT EXCERPTS ---
{context_text}
--- END EXCERPTS ---

Answer the question using the excerpts above. Be concise (2-3 sentences)."""
    else:
        augmented_message = user_message

    session["conversation"].append({"role": "user", "content": user_message})
    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(session["conversation"][-20:-1])
    messages.append({"role": "user", "content": augmented_message})

    response = ollama.chat(model=MODEL, messages=messages)
    reply = response["message"]["content"]
    session["conversation"].append({"role": "assistant", "content": reply})
    return reply


def reset_session():
    """Clear conversation history."""
    session["conversation"] = []
