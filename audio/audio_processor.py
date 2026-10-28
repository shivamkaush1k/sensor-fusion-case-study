from pathlib import Path
import wave

import numpy as np


class AudioProcessor:
    """Expose time-aligned, fixed-duration PCM chunks from a WAV file."""

    def __init__(self, audio_path=None, chunk_seconds=1.0):
        if chunk_seconds <= 0:
            raise ValueError("chunk_seconds must be positive")
        self.audio_path = Path(audio_path) if audio_path else None
        self.chunk_seconds = float(chunk_seconds)
        self.signal = np.array([], dtype=np.float32)
        self.sample_rate = 0

    def load_audio(self):
        if self.audio_path is None:
            return self.signal, self.sample_rate
        if not self.audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {self.audio_path}")
        with wave.open(str(self.audio_path), "rb") as wav:
            channels, width, rate, frames = wav.getnchannels(), wav.getsampwidth(), wav.getframerate(), wav.getnframes()
            if width not in (1, 2, 4):
                raise ValueError("Only 8-, 16-, and 32-bit PCM WAV files are supported")
            dtype = {1: np.uint8, 2: np.int16, 4: np.int32}[width]
            samples = np.frombuffer(wav.readframes(frames), dtype=dtype).astype(np.float32)
        if width == 1:
            samples = (samples - 128.0) / 128.0
        else:
            samples /= float(2 ** (8 * width - 1))
        if len(samples) == 0:
            self.signal = np.array([], dtype=np.float32)
        else:
            self.signal = samples.reshape(-1, channels).mean(axis=1) if channels > 1 else samples
        self.sample_rate = rate
        return self.signal, self.sample_rate

    def chunk_at(self, timestamp_seconds):
        """Return the audio window corresponding to a video timestamp."""
        if self.sample_rate == 0:
            return np.array([], dtype=np.float32)
        size = max(1, int(self.chunk_seconds * self.sample_rate))
        start = int(timestamp_seconds * self.sample_rate)
        chunk = self.signal[start : start + size]
        return chunk if len(chunk) else np.array([], dtype=np.float32)

