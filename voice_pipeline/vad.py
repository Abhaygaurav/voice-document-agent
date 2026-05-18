"""
Voice Activity Detection (VAD) Module
Uses Silero VAD to detect when the user starts/stops speaking.
Enables barge-in (interrupting the bot mid-sentence).
"""

import numpy as np
import torch

torch.set_num_threads(1)


class SileroVAD:
    """Voice Activity Detection using Silero VAD."""

    def __init__(self, threshold: float = 0.5, sample_rate: int = 16000):
        """
        Initialize Silero VAD.

        Args:
            threshold: Confidence threshold (0.0-1.0)
            sample_rate: Audio sample rate (16000 or 8000)
        """
        self.threshold = threshold
        self.sample_rate = sample_rate

        print("[VAD] Loading Silero model...")
        self.model, _ = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            trust_repo=True,
        )
        self.model.eval()
        print("[VAD] Model loaded.")

        self.is_speaking = False
        self._speech_start_callback = None
        self._speech_end_callback = None

    def on_speech_start(self, callback):
        """Register callback for when user starts speaking."""
        self._speech_start_callback = callback

    def on_speech_end(self, callback):
        """Register callback for when user stops speaking."""
        self._speech_end_callback = callback

    def process_chunk(self, audio_chunk: np.ndarray) -> float:
        """
        Process an audio chunk and return speech probability.

        Args:
            audio_chunk: numpy array of audio samples (float32, 512 samples)

        Returns:
            Speech probability (0.0 to 1.0)
        """
        tensor = torch.from_numpy(audio_chunk).float()
        speech_prob = self.model(tensor, self.sample_rate).item()

        was_speaking = self.is_speaking

        if speech_prob >= self.threshold and not was_speaking:
            self.is_speaking = True
            if self._speech_start_callback:
                self._speech_start_callback()
        elif speech_prob < self.threshold * 0.8 and was_speaking:
            self.is_speaking = False
            if self._speech_end_callback:
                self._speech_end_callback()

        return speech_prob

    def reset(self):
        """Reset VAD state."""
        self.model.reset_states()
        self.is_speaking = False


if __name__ == "__main__":
    vad = SileroVAD(threshold=0.5)
    vad.on_speech_start(lambda: print("🎤 Speech started"))
    vad.on_speech_end(lambda: print("🔇 Speech ended"))

    silence = np.zeros(512, dtype=np.float32)
    prob = vad.process_chunk(silence)
    print(f"Silence probability: {prob:.3f}")
    print("VAD ready.")
