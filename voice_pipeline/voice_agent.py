"""
Voice Agent — Terminal-based voice conversation with your document.
Combines STT + TTS + VAD + Ollama for a hands-free experience.

Usage: python -m voice_pipeline.voice_agent [path_to_summary.json]
"""

import numpy as np
import time
import json
import sys
import os

try:
    import sounddevice as sd
    HAS_AUDIO = True
except ImportError:
    HAS_AUDIO = False

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from voice_pipeline.stt import LocalSTT
from voice_pipeline.tts import LocalTTS
from voice_pipeline.vad import SileroVAD


class VoiceAgent:
    """Terminal voice agent for document Q&A."""

    def __init__(self, summary_path: str = None):
        print("=" * 50)
        print("  Voice Document Agent (Terminal)")
        print("=" * 50)

        self.stt = LocalSTT(model_size="base")
        self.tts = LocalTTS(rate=175)
        self.vad = SileroVAD(threshold=0.5)

        self.summary = ""
        if summary_path and os.path.exists(summary_path):
            with open(summary_path) as f:
                data = json.load(f)
            self.summary = data.get("summary", "")
            print(f"[Agent] Loaded summary ({len(self.summary)} chars)")

        self._running = False
        self._audio_buffer = []
        self.is_listening = False

        self.vad.on_speech_start(self._on_speech_start)
        self.vad.on_speech_end(self._on_speech_end)

        print("\n✅ All components loaded.")

    def _on_speech_start(self):
        self.is_listening = True
        self._audio_buffer = []

    def _on_speech_end(self):
        self.is_listening = False
        if not self._audio_buffer:
            return

        audio = np.concatenate(self._audio_buffer)
        self._audio_buffer = []

        print("\n🎤 Transcribing...")
        text = self.stt.transcribe(audio)
        if not text.strip():
            return

        print(f"   You: {text}")
        print("🤔 Thinking...")

        # Call the backend
        try:
            import httpx
            with httpx.Client() as client:
                resp = client.post(
                    "http://localhost:8000/chat",
                    json={"message": text},
                    timeout=30.0,
                )
                reply = resp.json().get("reply", "Sorry, I couldn't process that.")
        except Exception as e:
            reply = f"Backend error: {e}"

        print(f"   Agent: {reply}")
        self.tts.speak(reply)

    def start(self):
        """Start listening loop."""
        if not HAS_AUDIO:
            print("ERROR: sounddevice not installed. Run: pip install sounddevice")
            return

        self._running = True
        print("\n🎧 Listening... (speak to ask questions, Ctrl+C to quit)\n")

        try:
            with sd.InputStream(
                samplerate=16000, channels=1, dtype="float32",
                blocksize=512, callback=self._audio_callback,
            ):
                while self._running:
                    time.sleep(0.1)
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")

    def _audio_callback(self, indata, frames, time_info, status):
        audio_chunk = indata[:, 0]
        self.vad.process_chunk(audio_chunk)
        if self.is_listening:
            self._audio_buffer.append(audio_chunk.copy())


if __name__ == "__main__":
    summary_path = sys.argv[1] if len(sys.argv) > 1 else "document_summary.json"
    agent = VoiceAgent(summary_path=summary_path)
    agent.start()
