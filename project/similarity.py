"""Audio similarity scoring using MFCC + DTW."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Optional deps: librosa, fastdtw
try:
    import librosa  # type: ignore
    from fastdtw import fastdtw  # type: ignore
    _HAS_SIM_DEPS = True
except Exception:
    librosa = None  # type: ignore
    fastdtw = None  # type: ignore
    _HAS_SIM_DEPS = False

from .config import AudioSettings


@dataclass(slots=True)
class SimilarityScorer:
    settings: AudioSettings

    def _load_mfcc(self, path: Path) -> np.ndarray:
        if not _HAS_SIM_DEPS:
            raise ImportError("Similarity scoring requires librosa and fastdtw.")
        audio, sr = librosa.load(path, sr=self.settings.sample_rate)
        return librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=20)

    def compare(self, reference: Path, attempt: Path) -> float:
        if not _HAS_SIM_DEPS:
            raise ImportError("Similarity scoring requires librosa and fastdtw.")
        ref_mfcc = self._load_mfcc(reference)
        att_mfcc = self._load_mfcc(attempt)
        distance, _ = fastdtw(ref_mfcc.T, att_mfcc.T)
        return 1.0 / (1.0 + distance)
