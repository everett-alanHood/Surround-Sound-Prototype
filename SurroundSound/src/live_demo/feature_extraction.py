from dataclasses import dataclass
from typing import Optional

import numpy as np
import librosa


@dataclass
class AudioFeatureExtractor:
    """
    Unified feature extractor for both environment and events models.

    Environment model uses VGGish-style embeddings (10x128) approximated
    from log-mel for live inference. Events model uses normalized log-mel.

      - 16 kHz mono
      - 128 mel bins
      - hop_length = 512
      - fmax = 8000
    """

    sample_rate: int = 16000
    n_mels: int = 128
    hop_length: int = 512
    fmin: float = 0.0
    fmax: float = 8000.0
    db_min: float = -80.0
    db_max: float = 0.0

    # VGGish embedding approximation config
    n_frames: int = 10      # 10 seconds = 10 embeddings at 1Hz
    embed_dim: int = 128

    def normalize_audio(self, y: np.ndarray) -> np.ndarray:
        if y.ndim > 1:
            y = np.mean(y, axis=-1)
        peak = np.max(np.abs(y)) if y.size > 0 else 0.0
        return (y / peak if peak > 0 else y).astype(np.float32)

    def _logmel(self, y: np.ndarray, sr: Optional[int] = None) -> np.ndarray:
        sr = sr or self.sample_rate
        mel = librosa.feature.melspectrogram(
            y=y, sr=sr, n_mels=self.n_mels,
            hop_length=self.hop_length, fmin=self.fmin, fmax=self.fmax, power=2.0,
        )
        return librosa.power_to_db(mel, ref=np.max).astype(np.float32)

    def env_features(self, audio: np.ndarray, sr: Optional[int] = None) -> np.ndarray:
        """
        Approximate VGGish-style embeddings for the environment model.
        Divides audio into 1-second windows, computes mean log-mel per window,
        producing a (10, 128) matrix matching the training input format.
        Returns (128, 10) for Conv1d input.
        """
        sr = sr or self.sample_rate
        y  = self.normalize_audio(audio)

        # Pad or trim to exactly 10 seconds
        target = sr * self.n_frames
        if len(y) < target:
            y = np.pad(y, (0, target - len(y)))
        else:
            y = y[:target]

        embeddings = []
        window = sr  # 1 second per embedding
        for i in range(self.n_frames):
            chunk  = y[i * window:(i + 1) * window]
            logmel = self._logmel(chunk, sr=sr)  # (128, T_chunk)
            # Mean pool over time -> (128,)
            emb = logmel.mean(axis=1)
            # Normalize to [-2, 2] to roughly match VGGish dequantized range
            emb = np.clip(emb / 40.0, -2.0, 2.0)
            embeddings.append(emb)

        emb_matrix = np.stack(embeddings, axis=0)  # (10, 128)
        return emb_matrix.T.astype(np.float32)      # (128, 10) for Conv1d

    def env_logmel(self, audio: np.ndarray, sr: Optional[int] = None) -> np.ndarray:
        """Full log-mel for display purposes only."""
        y = self.normalize_audio(audio)
        return self._logmel(y, sr=sr)

    def event_features(self, audio: np.ndarray, sr: Optional[int] = None) -> np.ndarray:
        """
        Normalized log-mel for the events model: dB -> [0, 1].
        Returns (128, T).
        """
        y      = self.normalize_audio(audio)
        logmel = self._logmel(y, sr=sr)
        spec   = np.clip((logmel - self.db_min) / (self.db_max - self.db_min), 0.0, 1.0)
        return spec.astype(np.float32)