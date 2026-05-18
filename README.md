# 🎙️ Voice Document Agent

Upload a PDF and talk to it. Fully local RAG pipeline with voice — runs on a laptop with zero API keys.

![Python](https://img.shields.io/badge/Python-3.11+-blue)
![Ollama](https://img.shields.io/badge/LLM-Ollama%20Llama%203.2-green)
![ChromaDB](https://img.shields.io/badge/VectorDB-ChromaDB-orange)

## What It Does

1. **Upload** any PDF document
2. **Get** an AI-generated executive summary instantly
3. **Ask questions** by voice or text — answers are grounded in the actual document (RAG)
4. **Listen** to the summary read aloud with pause/resume controls
5. **Interrupt** mid-sentence to ask a follow-up question

Everything runs locally on your machine. No cloud APIs, no data leaves your computer.

## Demo

```
Upload PDF → AI reads & chunks it → Embeds in vector DB → Generates summary
     ↓
Ask "What's the notice period?" (voice or text)
     ↓
Semantic search finds relevant chunks → LLM answers with exact quotes
```

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     BROWSER (Chrome)                         │
│  Left: Q&A History │ Center: Upload/Voice │ Right: Chat     │
│  Web Speech API (STT) + SpeechSynthesis (TTS)               │
└────────────────────────────┬────────────────────────────────┘
                             │ HTTP
                             ▼
┌─────────────────────────────────────────────────────────────┐
│                  PYTHON BACKEND (FastAPI)                     │
│                                                              │
│  POST /upload → PyMuPDF → Chunk → Embed → ChromaDB          │
│                         → Summarize via Ollama               │
│                                                              │
│  POST /chat → Embed question → Search ChromaDB (top 4)      │
│             → Feed chunks + summary to Ollama → reply        │
└────────────────────────────┬────────────────────────────────┘
                             │
              ┌──────────────┼──────────────┐
              ▼              ▼              ▼
        ┌──────────┐  ┌──────────────┐  ┌──────────┐
        │  Ollama  │  │ nomic-embed  │  │ ChromaDB │
        │ llama3.2 │  │    -text     │  │  (local) │
        │  (chat)  │  │ (embeddings) │  │          │
        └──────────┘  └──────────────┘  └──────────┘
```

## Setup

```bash
# 1. Clone
git clone https://github.com/YOUR_USERNAME/voice-document-agent.git
cd voice-document-agent

# 2. Install dependencies
pip install -r requirements.txt

# 3. Install Ollama
brew install ollama    # macOS
# See https://ollama.com for other platforms

# 4. Pull models
ollama pull llama3.2
ollama pull nomic-embed-text

# 5. Start Ollama (separate terminal)
ollama serve
```

## Run

```bash
python app.py
```

Open **http://localhost:8000** in Chrome.

## Features

| Feature | How |
|---------|-----|
| PDF parsing | PyMuPDF — handles complex layouts |
| Smart chunking | 1000-char chunks with 200-char overlap |
| Vector search (RAG) | ChromaDB + nomic-embed-text embeddings |
| Summary generation | Ollama llama3.2 |
| Voice input | Chrome Web Speech API |
| Voice output | Browser SpeechSynthesis with pause/resume |
| Barge-in | Interrupt the bot mid-sentence |
| Session history | Left sidebar tracks all Q&A |
| Download summary | Export as text file |

## How RAG Works

Without RAG, the LLM only knows a brief summary and hallucinates on specific questions.

With RAG:
1. Every chunk is embedded (768-dim vector via nomic-embed-text)
2. Stored in ChromaDB for instant similarity search
3. When you ask a question, it's embedded and matched against all chunks
4. The 4 most relevant chunks are fed to the LLM alongside your question
5. The LLM answers using **actual document text**, not guesses

## Project Structure

```
voice-document-agent/
├── app.py                     # Web app (FastAPI backend + HTML frontend)
├── requirements.txt           # Core dependencies
├── README.md
├── .gitignore
└── voice_pipeline/            # Modular voice processing components
    ├── __init__.py
    ├── stt.py                 # Speech-to-Text (faster-whisper, local)
    ├── tts.py                 # Text-to-Speech (pyttsx3, local)
    ├── vad.py                 # Voice Activity Detection (Silero VAD)
    ├── audio_player.py        # Audio playback with barge-in support
    ├── voice_agent.py         # Terminal voice agent (combines all modules)
    └── README.md
```

## Tech Stack

| Component | Tool |
|-----------|------|
| Web server | FastAPI + Uvicorn |
| PDF parsing | PyMuPDF |
| Text splitting | LangChain RecursiveCharacterTextSplitter |
| Embeddings | Ollama + nomic-embed-text (768-dim, local) |
| Vector DB | ChromaDB (in-memory, local) |
| LLM | Ollama + Llama 3.2 (3B params, runs on GPU) |
| Voice (web) | Chrome Web Speech API + SpeechSynthesis |
| Voice (terminal) | faster-whisper + pyttsx3 + Silero VAD |

## Requirements

- macOS / Linux 
- Python 3.11+
- Ollama
- Chrome (for voice features)

## Limitations

- Single user (in-memory session)
- Summary uses first 5 chunks (very long docs may have incomplete overview)
- Voice features require Chrome
- No persistence (restarting clears data)

## License

MIT
