# Voice Pipeline — Local Voice Modules

Standalone voice processing components. Can be used independently or with the web app.

## Modules

| File | Purpose | Dependency |
|------|---------|-----------|
| `stt.py` | Speech-to-Text | faster-whisper |
| `tts.py` | Text-to-Speech | pyttsx3 |
| `vad.py` | Voice Activity Detection (barge-in) | torch, silero-vad |
| `audio_player.py` | Audio playback with interrupt | sounddevice |
| `voice_agent.py` | Terminal voice agent (combines all) | all above + httpx |

## Terminal Voice Agent

Talk to your document from the terminal (no browser needed):

```bash
# Make sure app.py is running (for the RAG backend)
python -m voice_pipeline.voice_agent document_summary.json
```

## Install Dependencies

```bash
pip install faster-whisper pyttsx3 torch sounddevice httpx
```
