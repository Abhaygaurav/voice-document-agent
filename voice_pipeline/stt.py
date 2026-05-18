"""
Speech-to-Text Module
Uses faster-whisper for local transcription (no API key needed).
"""

from faster_whisper import WhisperModel
import numpy as np


class LocalSTT:
    """Local Speech-to-Text using faster-whisper."""

    def __init__(self, model_size: str = "base", device: str = "cpu"):
        """
        Initialize the STT model.

        Args:
            model_size: "tiny", "base", "small", "medium"
            device: "cpu" or "cuda"
        """
        print(f"[STT] Loading Whisper model ({model_size})...")
        self.model = WhisperModel(model_size, device=device, compute_type="int8")
        print("[STT] Model loaded.")

    def transcribe(self, audio_data: np.ndarray, sample_rate: int = 16000) -> str:
        """
        Transcribe audio to text.

        Args:
            audio_data: numpy array of audio samples (float32, mono)
            sample_rate: sample rate of the audio

        Returns:
            Transcribed text string
        """
        segments, info = self.model.transcribe(
            audio_data, beam_size=5, language="en", vad_filter=True,
        )
        return " ".join(seg.text.strip() for seg in segments)

    def transcribe_file(self, audio_path: str) -> str:
        """Transcribe an audio file."""
        segments, info = self.model.transcribe(
            audio_path, beam_size=5, language="en", vad_filter=True,
        )
        return " ".join(seg.text.strip() for seg in segments)


if __name__ == "__main__":
    stt = LocalSTT(model_size="base")
    print("STT ready. Test with: stt.transcribe_file('test.wav')")
