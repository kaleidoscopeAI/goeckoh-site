from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Tuple

import librosa
import numpy as np
import pyttsx3
import soundfile as sf

from .config import SpeechModelSettings


def _ensure_mono(wav: np.ndarray) -> np.ndarray:
    if wav.ndim == 1:
        return wav
    return np.mean(wav, axis=1)


