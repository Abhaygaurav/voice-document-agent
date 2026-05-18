"""
Voice Document Agent
Upload a PDF → Get an AI summary → Ask questions by voice or text.
Fully local: Ollama + ChromaDB + Browser APIs. No API keys needed.

Run: python app.py
Open: http://localhost:8000 (Chrome recommended for voice)
"""

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse
import fitz  # PyMuPDF
from langchain_text_splitters import RecursiveCharacterTextSplitter
import ollama
import chromadb
import os
import tempfile

app = FastAPI(title="Voice Document Agent")

MODEL = "llama3.2"
EMBED_MODEL = "nomic-embed-text"

# ChromaDB client
chroma_client = chromadb.Client()
collection = None

session = {
    "summary": "",
    "chunks": [],
    "conversation": [],
}


# ─── INGESTION ──────────────────────────────────────────────────────────────────

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
        chunk_size: Size of each chunk in characters (default 1000)
        chunk_overlap: Overlap between chunks in characters (default 200)
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap, length_function=len,
    )
    return splitter.split_text(text)


def summarize_chunks(chunks: list[str]) -> str:
    """Generate an executive summary from the first chunks."""
    combined_text = "\n\n".join(chunks[:5])
    prompt = f"""You are a document analysis assistant. Below is the content of a document.
Please provide a concise executive summary that includes:
1. What type of document this is
2. The key parties involved
3. Main obligations and terms
4. Important dates or deadlines
5. Any critical clauses or conditions

Keep the summary clear and structured so it can be used as a briefing.

--- DOCUMENT CONTENT ---
{combined_text}
--- END OF CONTENT ---

Executive Summary:"""
    response = ollama.chat(model=MODEL, messages=[{"role": "user", "content": prompt}])
    return response["message"]["content"]


# ─── RAG CHAT ───────────────────────────────────────────────────────────────────

def chat_with_document(user_message: str) -> str:
    """Answer questions using RAG — retrieves relevant chunks from ChromaDB."""
    global collection
    system_prompt = f"""You are a helpful voice assistant that has analyzed a document.
You speak naturally and concisely — keep responses to 2-3 sentences since this is voice.
Use the document summary AND the relevant excerpts below to answer questions.
If the answer is in the excerpts, quote or reference the specific text.
If you cannot find the answer in the provided text, say so honestly.

--- DOCUMENT SUMMARY ---
{session['summary']}
--- END SUMMARY ---"""

    context_text = ""
    if collection is not None:
        try:
            query_embedding = ollama.embed(model=EMBED_MODEL, input=user_message)
            results = collection.query(
                query_embeddings=[query_embedding["embeddings"][0]], n_results=4,
            )
            if results and results["documents"] and results["documents"][0]:
                retrieved_chunks = results["documents"][0]
                context_text = "\n\n".join(
                    f"[Excerpt {i+1}]: {chunk}" for i, chunk in enumerate(retrieved_chunks)
                )
        except Exception as e:
            print(f"RAG search error: {e}")

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


# ─── API ROUTES ──────────────────────────────────────────────────────────────────

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    global collection
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name
    try:
        raw_text = extract_text_from_pdf(tmp_path)
        
        # Auto-adjust chunk size based on document length
        doc_len = len(raw_text)
        if doc_len < 5000:          # Short doc (~2-3 pages)
            chunk_size, overlap = 500, 100
        elif doc_len < 20000:       # Medium doc (~5-10 pages)
            chunk_size, overlap = 800, 150
        elif doc_len < 50000:       # Long doc (~15-25 pages)
            chunk_size, overlap = 1000, 200
        else:                       # Very long doc (25+ pages)
            chunk_size, overlap = 1500, 300

        chunks = chunk_text(raw_text, chunk_size=chunk_size, chunk_overlap=overlap)
        summary = summarize_chunks(chunks)
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
        session["summary"] = summary
        session["chunks"] = chunks
        session["conversation"] = []
        return {"source_file": file.filename, "total_characters": len(raw_text),
                "total_chunks": len(chunks), "summary": summary}
    finally:
        os.unlink(tmp_path)


@app.post("/chat")
async def chat_endpoint(data: dict):
    user_message = data.get("message", "")
    if not session["summary"]:
        return {"reply": "Please upload a document first."}
    return {"reply": chat_with_document(user_message)}


@app.get("/", response_class=HTMLResponse)
async def home():
    return HTML_PAGE


# ─── FRONTEND ────────────────────────────────────────────────────────────────────

HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>DocTalk — Voice Document Agent</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
<script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
<style>
:root {
    --bg: #0c0c14;
    --surface: #13131f;
    --surface-2: #1a1a2e;
    --border: #252540;
    --text: #e8e8f0;
    --text-muted: #8888aa;
    --primary: #6c5ce7;
    --primary-light: #a29bfe;
    --accent: #00cec9;
    --accent-light: #81ecec;
    --success: #00b894;
    --warning: #fdcb6e;
    --danger: #ff6b6b;
    --gradient: linear-gradient(135deg, #6c5ce7 0%, #00cec9 100%);
    --radius: 12px;
    --shadow: 0 4px 24px rgba(0,0,0,0.3);
}
* { margin: 0; padding: 0; box-sizing: border-box; }
body { font-family: 'Inter', sans-serif; background: var(--bg); color: var(--text); height: 100vh; display: flex; overflow: hidden; }

/* Left Sidebar */
.sidebar {
    width: 280px; background: var(--surface); border-right: 1px solid var(--border);
    display: flex; flex-direction: column; flex-shrink: 0;
}
.sidebar-header { padding: 24px 20px 16px; border-bottom: 1px solid var(--border); }
.sidebar-header h2 { font-size: 0.75rem; font-weight: 600; color: var(--text-muted); text-transform: uppercase; letter-spacing: 1.5px; }
.sidebar-content { flex: 1; overflow-y: auto; padding: 16px; }
.sidebar-empty { color: #444466; font-size: 0.82rem; text-align: center; padding: 32px 16px; line-height: 1.6; }
.qa-item { margin-bottom: 12px; padding: 14px; background: var(--surface-2); border-radius: var(--radius); border: 1px solid var(--border); transition: all 0.2s; cursor: pointer; }
.qa-item:hover { border-color: var(--primary); transform: translateY(-1px); box-shadow: 0 4px 12px rgba(108,92,231,0.1); }
.qa-item .q { font-size: 0.8rem; color: var(--primary-light); font-weight: 500; margin-bottom: 6px; }
.qa-item .a { font-size: 0.75rem; color: var(--text-muted); line-height: 1.5; }
.qa-item .time { font-size: 0.65rem; color: #444466; margin-top: 6px; }
</style>
</head>
<body>
<div class="sidebar">
    <div class="sidebar-header"><h2>Session History</h2></div>
    <div class="sidebar-content" id="sidebarContent">
        <p class="sidebar-empty">💡 Your Q&A history will<br>appear here as you chat<br><br><span style="font-size:0.72rem;color:#333355;">Try asking:<br>"What are the key terms?"<br>"Summarize section 3"</span></p>
    </div>
</div>
<style>
/* Center Panel */
.main { flex: 1; display: flex; flex-direction: column; overflow-y: auto; padding: 40px; }
.hero { margin-bottom: 32px; }
.hero h1 { font-size: 2rem; font-weight: 700; background: var(--gradient); -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin-bottom: 6px; }
.hero p { color: var(--text-muted); font-size: 0.9rem; font-weight: 300; }

/* Upload Card */
.upload-card { background: var(--surface); border: 1px solid var(--border); border-radius: 16px; padding: 28px; margin-bottom: 24px; transition: all 0.3s; }
.upload-card:hover { border-color: var(--primary); box-shadow: 0 0 30px rgba(108,92,231,0.08); }
.upload-card.collapsed { padding: 14px 20px; }
.upload-card.collapsed .drop-zone { display: none; }
.upload-card.collapsed .btn-primary { display: none; }
.upload-card.collapsed .loading { display: none; }
.file-header { display: none; align-items: center; justify-content: space-between; }
.upload-card.collapsed .file-header { display: flex; }
.file-header .file-info { display: flex; align-items: center; gap: 10px; font-size: 0.84rem; color: var(--accent); font-weight: 500; }
.file-header .btn-new { background: rgba(108,92,231,0.1); color: var(--primary-light); border: 1px solid rgba(108,92,231,0.3); padding: 6px 14px; border-radius: 8px; font-size: 0.75rem; font-weight: 500; cursor: pointer; }
.file-header .btn-new:hover { background: rgba(108,92,231,0.2); }
.drop-zone { border: 2px dashed var(--border); border-radius: var(--radius); padding: 36px; text-align: center; cursor: pointer; transition: all 0.3s; margin-bottom: 16px; }
.drop-zone:hover, .drop-zone.dragover { border-color: var(--primary); background: rgba(108,92,231,0.05); }
.drop-zone .icon { font-size: 2.4rem; margin-bottom: 8px; filter: grayscale(0.3); }
.drop-zone p { color: var(--text-muted); font-size: 0.85rem; }
.drop-zone .file-name { color: var(--accent); font-size: 0.82rem; margin-top: 8px; font-weight: 500; }
input[type="file"] { display: none; }
.btn-primary { background: var(--gradient); color: #fff; border: none; padding: 12px 24px; border-radius: 10px; font-size: 0.88rem; font-weight: 600; cursor: pointer; width: 100%; transition: all 0.3s; letter-spacing: 0.3px; }
.btn-primary:hover { opacity: 0.9; transform: translateY(-1px); box-shadow: 0 6px 20px rgba(108,92,231,0.3); }
.btn-primary:disabled { opacity: 0.3; cursor: not-allowed; transform: none; box-shadow: none; }

/* Summary Card */
.summary-card { display: none; background: var(--surface); border: 1px solid var(--border); border-radius: 16px; padding: 24px; margin-bottom: 24px; }
.summary-card .header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 14px; }
.summary-card .header h2 { font-size: 1rem; font-weight: 600; color: var(--accent); }
.btn-download { background: rgba(0,206,201,0.1); color: var(--accent); border: 1px solid rgba(0,206,201,0.3); padding: 6px 14px; border-radius: 8px; font-size: 0.75rem; font-weight: 500; cursor: pointer; transition: all 0.2s; }
.btn-download:hover { background: rgba(0,206,201,0.2); }
.summary-text { font-size: 0.84rem; line-height: 1.8; color: #bbbbd0; max-height: 280px; overflow-y: auto; padding-right: 8px; }
.summary-text h1,.summary-text h2,.summary-text h3 { color: var(--accent-light); margin: 12px 0 6px; font-size: 0.92rem; }
.summary-text strong { color: var(--text); }
.summary-text ul,.summary-text ol { padding-left: 20px; margin: 6px 0; }
.summary-text li { margin-bottom: 4px; }
.summary-text p { margin-bottom: 8px; }
.summary-text::-webkit-scrollbar { width: 4px; }
.summary-text::-webkit-scrollbar-thumb { background: var(--border); border-radius: 4px; }

/* Voice Controls — sticky at bottom */
.voice-card { display: none; background: var(--surface); border: 1px solid var(--border); border-radius: 16px; padding: 24px; position: sticky; bottom: 16px; z-index: 10; box-shadow: 0 -4px 20px rgba(0,0,0,0.3); }
.voice-card h2 { font-size: 1rem; font-weight: 600; margin-bottom: 16px; }
.voice-row { display: flex; align-items: center; gap: 12px; flex-wrap: wrap; }
.mic-btn { width: 56px; height: 56px; border-radius: 50%; border: 2px solid var(--primary); background: rgba(108,92,231,0.1); color: var(--primary-light); font-size: 1.4rem; cursor: pointer; transition: all 0.3s; display: flex; align-items: center; justify-content: center; }
.mic-btn:hover { background: rgba(108,92,231,0.2); transform: scale(1.05); box-shadow: 0 0 20px rgba(108,92,231,0.3); }
.mic-btn.recording { border: none; background: var(--gradient); color: #fff; animation: pulse 1.5s infinite; box-shadow: 0 0 24px rgba(108,92,231,0.5); }
@keyframes pulse { 0%,100%{box-shadow:0 0 0 0 rgba(255,107,107,0.4)} 50%{box-shadow:0 0 0 12px rgba(255,107,107,0)} }
.ctrl { padding: 8px 14px; border-radius: 8px; font-size: 0.78rem; font-weight: 500; cursor: pointer; border: 1px solid; transition: all 0.2s; }
.ctrl-speak { background: rgba(0,184,148,0.1); color: var(--success); border-color: rgba(0,184,148,0.3); }
.ctrl-speak:hover { background: rgba(0,184,148,0.2); }
.ctrl-pause { background: rgba(253,203,110,0.1); color: var(--warning); border-color: rgba(253,203,110,0.3); display: none; }
.ctrl-resume { background: rgba(0,184,148,0.1); color: var(--success); border-color: rgba(0,184,148,0.3); display: none; }
.ctrl-stop { background: rgba(255,107,107,0.1); color: var(--danger); border-color: rgba(255,107,107,0.3); display: none; }
.voice-status { font-size: 0.78rem; color: var(--text-muted); margin-left: 4px; }
.shortcuts-hint { font-size: 0.7rem; color: #444466; margin-top: 12px; letter-spacing: 0.3px; }
.stats-badge { font-size: 0.7rem; background: var(--surface-2); color: var(--text-muted); padding: 4px 10px; border-radius: 20px; border: 1px solid var(--border); }
.loading { display: none; align-items: center; gap: 8px; margin-top: 12px; font-size: 0.78rem; color: var(--text-muted); }
.loading.active { display: flex; }

/* Light theme */
body.light { --bg: #f5f5fa; --surface: #ffffff; --surface-2: #f0f0f8; --border: #e0e0ee; --text: #1a1a2e; --text-muted: #666688; }
body.light .sidebar { background: #fafafe; }
body.light .msg.agent .bubble { background: #f0f0f8; color: #1a1a2e; }
body.light .chat-input input { background: #f5f5fa; color: #1a1a2e; }
body.light .drop-zone { border-color: #d0d0e0; }
body.light .summary-text { color: #333355; }
.spinner { width: 14px; height: 14px; border: 2px solid var(--border); border-top-color: var(--primary); border-radius: 50%; animation: spin 0.7s linear infinite; }
@keyframes spin { to { transform: rotate(360deg); } }
</style>

<div class="main">
    <div class="hero">
        <h1>DocTalk</h1>
        <p>Upload a document, get instant insights, ask anything by voice</p>
    </div>

    <div class="upload-card" id="uploadCard">
        <div class="file-header">
            <div class="file-info">📄 <span id="uploadedFileName"></span></div>
            <button class="btn-new" onclick="resetUpload()">Upload New</button>
        </div>
        <div class="drop-zone" id="dropZone" onclick="document.getElementById('fileInput').click()">
            <div class="icon">📄</div>
            <p>Drop your PDF here or click to browse</p>
            <div class="file-name" id="fileName"></div>
        </div>
        <button class="btn-primary" id="uploadBtn" onclick="uploadFile()" disabled>Analyze Document</button>
        <div class="loading" id="uploadLoading"><div class="spinner"></div><span>Analyzing with Llama 3.2 — this may take a moment...</span></div>
    </div>
    <input type="file" id="fileInput" accept=".pdf">

    <div class="summary-card" id="summarySection">
        <div class="header">
            <h2>Executive Summary</h2>
            <div style="display:flex;gap:8px;align-items:center;">
                <span class="stats-badge" id="statsBadge"></span>
                <button class="btn-download" onclick="copySummary()" title="Copy to clipboard">📋 Copy</button>
                <button class="btn-download" onclick="downloadSummary()">⬇ Download</button>
            </div>
        </div>
        <div class="summary-text" id="summaryText"></div>
    </div>

    <div class="voice-card" id="voiceSection">
        <h2>Voice Controls</h2>
        <div class="voice-row">
            <button class="mic-btn" id="micBtn" onclick="toggleRecording()">🎤</button>
            <button class="ctrl ctrl-speak" onclick="speakSummary()">🔊 Read Aloud</button>
            <select class="ctrl ctrl-speak" id="speedSelect" onchange="setSpeed(this.value)" title="Speech speed">
                <option value="0.75">0.75x</option>
                <option value="1" selected>1x</option>
                <option value="1.25">1.25x</option>
                <option value="1.5">1.5x</option>
                <option value="2">2x</option>
            </select>
            <select class="ctrl ctrl-speak" id="voiceSelect" onchange="setVoice(this.value)" title="Choose voice">
                <option value="">Loading voices...</option>
            </select>
            <button class="ctrl ctrl-pause" id="pauseBtn" onclick="pauseSpeaking()">⏸ Pause</button>
            <button class="ctrl ctrl-resume" id="resumeBtn" onclick="resumeSpeaking()">▶ Resume</button>
            <button class="ctrl ctrl-stop" id="stopBtn" onclick="stopSpeakingBtn()">⏹ Stop</button>
            <span class="voice-status" id="voiceStatus">Click mic to ask a question</span>
        </div>
        <div class="shortcuts-hint">⌨️ Space = mic · Esc = stop · T = theme</div>
        <div class="loading" id="voiceLoading"><div class="spinner"></div><span id="voiceLoadingText">Processing...</span></div>
    </div>
</div>
<style>
/* Right Chat Panel */
.chat-panel { width: 360px; background: var(--surface); border-left: 1px solid var(--border); display: none; flex-direction: column; flex-shrink: 0; }
.chat-panel.active { display: flex; }
.chat-panel .header { padding: 20px; border-bottom: 1px solid var(--border); display: flex; justify-content: space-between; align-items: center; }
.chat-panel .header h2 { font-size: 0.95rem; font-weight: 600; }
.chat-panel .close-btn { background: none; border: none; color: var(--text-muted); font-size: 1.3rem; cursor: pointer; padding: 4px 8px; border-radius: 6px; }
.chat-panel .close-btn:hover { background: var(--surface-2); color: var(--text); }
.chat-messages { flex: 1; overflow-y: auto; padding: 20px; }
.chat-messages::-webkit-scrollbar { width: 4px; }
.chat-messages::-webkit-scrollbar-thumb { background: var(--border); border-radius: 4px; }
.msg { margin-bottom: 16px; max-width: 88%; animation: fadeIn 0.3s ease; }
@keyframes fadeIn { from { opacity: 0; transform: translateY(8px); } to { opacity: 1; transform: translateY(0); } }
.msg.user { margin-left: auto; }
.msg .bubble { padding: 12px 16px; border-radius: 16px; font-size: 0.84rem; line-height: 1.6; }
.msg.user .bubble { background: var(--primary); color: #fff; border-bottom-right-radius: 4px; }
.msg.agent .bubble { background: var(--surface-2); color: var(--text); border: 1px solid var(--border); border-bottom-left-radius: 4px; }
.msg .label { font-size: 0.68rem; font-weight: 600; margin-bottom: 4px; padding: 0 4px; display: flex; justify-content: space-between; align-items: center; }
.msg.user .label { color: var(--primary-light); text-align: right; justify-content: flex-end; }
.msg.agent .label { color: var(--accent); }
.copy-btn { cursor: pointer; opacity: 0.4; transition: opacity 0.2s; font-size: 0.72rem; margin-left: 8px; }
.copy-btn:hover { opacity: 1; }
.chat-input { padding: 16px 20px; border-top: 1px solid var(--border); display: flex; gap: 10px; }
.chat-input input { flex: 1; background: var(--surface-2); border: 1px solid var(--border); border-radius: 10px; padding: 12px 16px; color: var(--text); font-size: 0.84rem; font-family: 'Inter', sans-serif; transition: border-color 0.2s; }
.chat-input input:focus { outline: none; border-color: var(--primary); }
.chat-input input::placeholder { color: #555577; }
.chat-input button { background: var(--gradient); color: #fff; border: none; padding: 12px 18px; border-radius: 10px; font-size: 0.82rem; font-weight: 600; cursor: pointer; transition: all 0.2s; }
.chat-input button:hover { opacity: 0.9; }
</style>

<div class="chat-panel" id="chatPanel">
    <div class="header">
        <h2>💬 Chat</h2>
        <div style="display:flex;gap:8px;align-items:center;">
            <button class="close-btn" onclick="clearConversation()" title="Clear conversation">🗑</button>
            <button class="close-btn" onclick="toggleTheme()" title="Toggle theme (T)">🌓</button>
            <button class="close-btn" onclick="toggleChat()">✕</button>
        </div>
    </div>
    <div class="chat-messages" id="chatMessages"></div>
    <div class="chat-input">
        <input type="text" id="chatInput" placeholder="Ask anything about the document..." onkeydown="if(event.key==='Enter')sendChat()">
        <button onclick="sendChat()">Send</button>
    </div>
</div>
<script>
let selectedFile=null,isRecording=false,currentSummary='',synth=window.speechSynthesis,speechPosition=0,isSpeakingSummary=false,qaHistory=[];
let speechRate = 1.0;
let selectedVoice = null;

function setSpeed(val) { 
    speechRate = parseFloat(val); 
    if (synth.speaking) {
        const wasSummary = isSpeakingSummary;
        synth.cancel();
        if (wasSummary && currentSummary) {
            isSpeakingSummary = true;
            speakText(currentSummary, speechPosition, () => { document.getElementById('voiceStatus').textContent = 'Summary complete'; });
        }
    }
}

function setVoice(idx) {
    const voices = synth.getVoices();
    selectedVoice = idx !== '' ? voices[parseInt(idx)] : null;
}

function populateVoices() {
    const voices = synth.getVoices();
    const sel = document.getElementById('voiceSelect');
    if (!voices.length) return;
    sel.innerHTML = '';
    voices.forEach((v, i) => {
        const opt = document.createElement('option');
        opt.value = i;
        const flag = v.lang.startsWith('en-IN') ? '🇮🇳 ' : v.lang.startsWith('en-US') ? '🇺🇸 ' : v.lang.startsWith('en-GB') ? '🇬🇧 ' : '';
        opt.textContent = flag + v.name.replace('com.apple.','') + ' (' + v.lang + ')';
        if (v.lang === 'en-IN') opt.style.fontWeight = '600';
        sel.appendChild(opt);
    });
    // Default to first en-IN or en-US voice
    const defaultIdx = voices.findIndex(v => v.lang === 'en-IN') !== -1 
        ? voices.findIndex(v => v.lang === 'en-IN') 
        : voices.findIndex(v => v.lang === 'en-US' && v.localService);
    if (defaultIdx >= 0) { sel.value = defaultIdx; selectedVoice = voices[defaultIdx]; }
}
const fileInput=document.getElementById('fileInput'),dropZone=document.getElementById('dropZone');
fileInput.addEventListener('change',e=>{selectedFile=e.target.files[0];if(selectedFile){document.getElementById('fileName').textContent=selectedFile.name;document.getElementById('uploadBtn').disabled=false;}});
dropZone.addEventListener('dragover',e=>{e.preventDefault();dropZone.classList.add('dragover');});
dropZone.addEventListener('dragleave',()=>dropZone.classList.remove('dragover'));
dropZone.addEventListener('drop',e=>{e.preventDefault();dropZone.classList.remove('dragover');selectedFile=e.dataTransfer.files[0];if(selectedFile){document.getElementById('fileName').textContent=selectedFile.name;document.getElementById('uploadBtn').disabled=false;}});

async function uploadFile(){
    if(!selectedFile)return;
    const btn=document.getElementById('uploadBtn'),loading=document.getElementById('uploadLoading');
    btn.disabled=true;loading.classList.add('active');
    const fd=new FormData();fd.append('file',selectedFile);
    try{const r=await fetch('/upload',{method:'POST',body:fd});const d=await r.json();
        currentSummary=d.summary;speechPosition=0;
        document.getElementById('summaryText').innerHTML=marked.parse(d.summary);
        document.getElementById('statsBadge').textContent=d.total_chunks+' chunks · '+Math.round(d.total_characters/1000)+'k chars';
        document.getElementById('summarySection').style.display='block';
        document.getElementById('voiceSection').style.display='block';
        document.getElementById('chatPanel').classList.add('active');
        // Collapse upload card
        document.getElementById('uploadCard').classList.add('collapsed');
        document.getElementById('uploadedFileName').textContent=d.source_file;
    }catch(e){alert('Error: '+e.message);}
    finally{loading.classList.remove('active');btn.disabled=false;}
}
function toggleChat(){document.getElementById('chatPanel').classList.toggle('active');}
function resetUpload(){
    document.getElementById('uploadCard').classList.remove('collapsed');
    document.getElementById('summarySection').style.display='none';
    document.getElementById('voiceSection').style.display='none';
    document.getElementById('fileName').textContent='';
    document.getElementById('uploadBtn').disabled=true;
    selectedFile=null; currentSummary='';
}
function downloadSummary(){if(!currentSummary)return;const b=new Blob([currentSummary],{type:'text/plain'});const a=document.createElement('a');a.href=URL.createObjectURL(b);a.download='document_summary.txt';a.click();}

function speakText(text,fromPos,onEnd){
    synth.cancel();
    var cleanText = text.substring(fromPos).replace(/[#*_`~>]/g, '');
    if(!cleanText.trim()){if(onEnd)onEnd();return;}
    var u=new SpeechSynthesisUtterance(cleanText);u.rate=speechRate;u.pitch=1.0;
    var voices=synth.getVoices();
    if(selectedVoice){u.voice=selectedVoice;}else if(voices.length){var v=voices.find(function(x){return x.lang==='en-IN';})||voices.find(function(x){return x.lang==='en-US'&&x.localService;})||voices[0];if(v)u.voice=v;}
    if(v)u.voice=v;
    u.onstart=()=>showCtrl(true);
    u.onboundary=e=>{if(e.name==='word')speechPosition=fromPos+e.charIndex;};
    u.onend=()=>{showCtrl(false);speechPosition=0;isSpeakingSummary=false;if(onEnd)onEnd();};
    u.onpause=()=>{document.getElementById('pauseBtn').style.display='none';document.getElementById('resumeBtn').style.display='inline-block';};
    u.onresume=()=>{document.getElementById('pauseBtn').style.display='inline-block';document.getElementById('resumeBtn').style.display='none';};
    synth.speak(u);
}
function speakSummary(){if(!currentSummary)return;synth.cancel();isSpeakingSummary=true;speakText(currentSummary,speechPosition,()=>{document.getElementById('voiceStatus').textContent='Summary complete';});}
function showCtrl(s){document.getElementById('pauseBtn').style.display=s?'inline-block':'none';document.getElementById('resumeBtn').style.display='none';document.getElementById('stopBtn').style.display=s?'inline-block':'none';}
function pauseSpeaking(){synth.pause();document.getElementById('voiceStatus').textContent='Paused — click Resume or ask a question';}
function resumeSpeaking(){if(synth.paused){synth.resume();document.getElementById('voiceStatus').textContent='Speaking...';}else if(isSpeakingSummary&&speechPosition>0){speakText(currentSummary,speechPosition,()=>{document.getElementById('voiceStatus').textContent='Summary complete';});}}
function stopSpeakingBtn(){synth.cancel();showCtrl(false);isSpeakingSummary=false;document.getElementById('voiceStatus').textContent='Stopped';}

let recognition=null;
function setupSR(){const SR=window.SpeechRecognition||window.webkitSpeechRecognition;if(!SR)return false;recognition=new SR();recognition.continuous=false;recognition.interimResults=false;recognition.lang='en-US';recognition.onresult=async e=>{await handleMsg(e.results[0][0].transcript);};recognition.onerror=()=>{document.getElementById('voiceStatus').textContent='Click mic to talk';document.getElementById('micBtn').classList.remove('recording');isRecording=false;};recognition.onend=()=>{document.getElementById('micBtn').classList.remove('recording');isRecording=false;};return true;}
function toggleRecording(){const m=document.getElementById('micBtn'),s=document.getElementById('voiceStatus');if(!recognition&&!setupSR()){s.textContent='Use Chrome for voice';return;}if(isRecording){recognition.stop();isRecording=false;m.classList.remove('recording');s.textContent='Processing...';}else{if(synth.speaking&&isSpeakingSummary){synth.pause();s.textContent='Summary paused — listening...';}else{synth.cancel();showCtrl(false);}recognition.start();isRecording=true;m.classList.add('recording');s.textContent='Listening... click to stop';}}
</script>
<script>
async function handleMsg(text){
    const s=document.getElementById('voiceStatus'),l=document.getElementById('voiceLoading');
    addMsg(text,'user');s.textContent='Thinking...';
    document.getElementById('voiceLoadingText').textContent='Generating response...';l.classList.add('active');
    try{const r=await fetch('/chat',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({message:text})});
        const d=await r.json();addMsg(d.reply,'agent');addSidebar(text,d.reply);s.textContent='Speaking...';
        speakText(d.reply,0,()=>{if(isSpeakingSummary&&speechPosition>0){s.textContent='Done — click Resume to continue';document.getElementById('resumeBtn').style.display='inline-block';document.getElementById('stopBtn').style.display='inline-block';}else{s.textContent='Click mic to ask another question';}});
    }catch(e){addMsg('Error: '+e.message,'agent');s.textContent='Error';}finally{l.classList.remove('active');}
}
async function sendChat(){const i=document.getElementById('chatInput');const t=i.value.trim();if(!t)return;i.value='';await handleMsg(t);}

function addMsg(text,role){
    const c=document.getElementById('chatMessages'),d=document.createElement('div');d.className='msg '+role;
    const copyBtn = role==='agent' ? '<span class="copy-btn" onclick="copyMsg(this)" title="Copy">📋</span>' : '';
    d.innerHTML='<div class="label">'+(role==='user'?'You':'DocTalk')+copyBtn+'</div><div class="bubble">'+text+'</div>';
    c.appendChild(d);c.scrollTop=c.scrollHeight;
}
function addSidebar(q,a){qaHistory.push({q,a,t:new Date()});renderSidebar();}
function renderSidebar(){
    const c=document.getElementById('sidebarContent');
    if(!qaHistory.length){c.innerHTML='<p class="sidebar-empty">Your conversations will<br>appear here as you chat</p>';return;}
    c.innerHTML=qaHistory.map((item,i)=>{
        const t=item.t.toLocaleTimeString([],{hour:'2-digit',minute:'2-digit'});
        const a=item.a.length>80?item.a.substring(0,80)+'...':item.a;
        return '<div class="qa-item"><div class="q">'+item.q+'</div><div class="a">'+a+'</div><div class="time">'+t+'</div></div>';
    }).reverse().join('');
}
speechSynthesis.onvoiceschanged=()=>{synth.getVoices();populateVoices();};
// Initial populate (some browsers load voices synchronously)
setTimeout(populateVoices, 100);

// ─── NEW FEATURES ───────────────────────────────────────────

function copySummary() {
    if (!currentSummary) return;
    navigator.clipboard.writeText(currentSummary).then(() => {
        const btn = event.target;
        const orig = btn.textContent;
        btn.textContent = '✓ Copied';
        setTimeout(() => btn.textContent = orig, 1500);
    });
}

function copyMsg(el) {
    const text = el.closest('.msg').querySelector('.bubble').textContent;
    navigator.clipboard.writeText(text).then(() => {
        el.textContent = '✓';
        setTimeout(() => el.textContent = '📋', 1000);
    });
}

function clearConversation() {
    if (!confirm('Clear all messages?')) return;
    document.getElementById('chatMessages').innerHTML = '';
    qaHistory = [];
    renderSidebar();
    fetch('/chat', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({message:'__clear__'})});
}

function toggleTheme() {
    document.body.classList.toggle('light');
    localStorage.setItem('theme', document.body.classList.contains('light') ? 'light' : 'dark');
}
// Load saved theme
if (localStorage.getItem('theme') === 'light') document.body.classList.add('light');

// Keyboard shortcuts
document.addEventListener('keydown', e => {
    // Don't trigger if typing in input
    if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;
    if (e.code === 'Space') { e.preventDefault(); toggleRecording(); }
    else if (e.code === 'Escape') { stopSpeakingBtn(); }
    else if (e.code === 'KeyT') { toggleTheme(); }
});
</script>
</body>
</html>
"""


if __name__ == "__main__":
    import uvicorn
    print("\\n✨ DocTalk — Voice Document Agent")
    print("📍 Open http://localhost:8000 in Chrome")
    print("   Fully local — no API keys needed\\n")
    uvicorn.run(app, host="0.0.0.0", port=8000)
