from dataclasses import dataclass
from typing import Optional

import numpy as np
import librosa


@dataclass
class AudioFeatureExtractor:
    """
    Unified feature extractor for both environment and events models.

    Matches your preprocessing:
      - 16 kHz mono
      - 128 mel bins
      - hop_length = 512
      - fmax = 8000
      - Environment: raw log-mel
      - Events: log-mel then normalized from [-80, 0] dB to [0, 1]
    """

    sample_rate: int = 16000
    n_mels: int = 128
    hop_length: int = 512
    fmin: float = 0.0
    fmax: float = 8000.0
    db_min: float = -80.0
    db_max: float = 0.0

    def normalize_audio(self, y: np.ndarray) -> np.ndarray:
        """Peak normalize."""
        if y.ndim > 1:
            y = np.mean(y, axis=-1)
        peak = np.max(np.abs(y)) if y.size > 0 else 0.0
        if peak > 0:
            y = y / peak
        return y.astype(np.float32)

    def _logmel(self, y: np.ndarray, sr: Optional[int] = None) -> np.ndarray:
        """Core log-mel computation shared by both models."""
        if sr is None:
            sr = self.sample_rate

        mel = librosa.feature.melspectrogram(
            y=y,
            sr=sr,
            n_mels=self.n_mels,
            hop_length=self.hop_length,
            fmin=self.fmin,
            fmax=self.fmax,
            power=2.0,
        )
        logmel = librosa.power_to_db(mel, ref=np.max)
        return logmel.astype(np.float32)

    # Environment model
    def env_features(self, audio: np.ndarray, sr: Optional[int] = None) -> np.ndarray:
        """
        Features for the environment model:
          - peak-normalized waveform
          - raw log-mel spectrogram
          - no DB scaling
        """
        y = self.normalize_audio(audio)
        logmel = self._logmel(y, sr=sr)
        return logmel.astype(np.float32)  # (n_mels, time)

    # Events model
    def event_features(self, audio: np.ndarray, sr: Optional[int] = None) -> np.ndarray:
        """
        Features for the events model:
          - peak-normalized waveform
          - log-mel
          - then map dB from [db_min, db_max] -> [0,1], clipped
        """
        y = self.normalize_audio(audio)
        logmel = self._logmel(y, sr=sr)

        spec = (logmel - self.db_min) / (self.db_max - self.db_min)
        spec = np.clip(spec, 0.0, 1.0)
        return spec.astype(np.float32)  # (n_mels, time)
