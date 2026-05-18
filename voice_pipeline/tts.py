"""
Text-to-Speech Module
Uses pyttsx3 for local TTS (no API key needed).
Works on macOS, Windows, and Linux.
"""

import io
import numpy as np


class LocalTTS:
    """Local Text-to-Speech using pyttsx3."""

    def __init__(self, rate: int = 175):
        """
        Initialize TTS engine.

        Args:
            rate: Speech speed (words per minute)
        """
        import pyttsx3
        self.engine = pyttsx3.init()
        self.engine.setProperty("rate", rate)
        self.rate = rate
        print("[TTS] Engine initialized.")

    def synthesize_to_file(self, text: str, output_path: str):
        """Generate speech and save to file."""
        self.engine.save_to_file(text, output_path)
        self.engine.runAndWait()
        print(f"[TTS] Audio saved to {output_path}")

    def speak(self, text: str):
        """Speak text directly through speakers."""
        self.engine.say(text)
        self.engine.runAndWait()


if __name__ == "__main__":
    tts = LocalTTS()
    tts.speak("Hello! I've analyzed your document. Here's a brief summary.")
    print("Done.")
