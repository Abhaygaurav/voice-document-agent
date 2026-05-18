"""
Audio Player Module
Plays audio with barge-in (interrupt) support.
"""

import numpy as np
import threading

try:
    import sounddevice as sd
    HAS_SOUNDDEVICE = True
except ImportError:
    HAS_SOUNDDEVICE = False


class AudioPlayer:
    """Plays audio with interrupt support."""

    def __init__(self, sample_rate: int = 24000):
        self.sample_rate = sample_rate
        self.is_playing = False
        self._stop_flag = threading.Event()
        self._play_thread = None

    def play(self, audio: np.ndarray, blocking: bool = False):
        """Play audio. Can be interrupted by calling stop()."""
        if not HAS_SOUNDDEVICE:
            print("[AudioPlayer] sounddevice not available")
            return

        self._stop_flag.clear()
        self.is_playing = True

        if blocking:
            self._play_blocking(audio)
        else:
            self._play_thread = threading.Thread(target=self._play_blocking, args=(audio,))
            self._play_thread.start()

    def _play_blocking(self, audio: np.ndarray):
        """Play audio in chunks for interruptibility."""
        chunk_size = self.sample_rate // 10  # 100ms chunks
        position = 0

        try:
            stream = sd.OutputStream(samplerate=self.sample_rate, channels=1, dtype="float32")
            stream.start()
            while position < len(audio) and not self._stop_flag.is_set():
                end = min(position + chunk_size, len(audio))
                stream.write(audio[position:end].reshape(-1, 1))
                position = end
            stream.stop()
            stream.close()
        except Exception as e:
            print(f"[AudioPlayer] Error: {e}")
        finally:
            self.is_playing = False

    def stop(self):
        """Stop playback immediately (barge-in)."""
        self._stop_flag.set()
        self.is_playing = False

    def wait(self):
        """Wait for playback to finish."""
        if self._play_thread and self._play_thread.is_alive():
            self._play_thread.join()


if __name__ == "__main__":
    if HAS_SOUNDDEVICE:
        player = AudioPlayer(sample_rate=24000)
        t = np.linspace(0, 2, 48000, dtype=np.float32)
        tone = 0.3 * np.sin(2 * np.pi * 440 * t)
        print("Playing 2s test tone...")
        player.play(tone, blocking=True)
        print("Done.")
    else:
        print("Install sounddevice: pip install sounddevice")
