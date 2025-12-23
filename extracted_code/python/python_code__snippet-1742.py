from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np

from .attempt_analysis import AttemptFeatures
from .voice_profile import VoiceProfile


def _sigmoid(x: float | np.ndarray) -> float | np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _smoothstep(x: float, edge0: float, edge1: float) -> float:
    x_norm = (x - edge0) / (edge1 - edge0 + 1e-8)
    x_norm = np.clip(x_norm, 0.0, 1.0)
    return float(x_norm * x_norm * (3.0 - 2.0 * x_norm))


def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return float(np.clip(x, lo, hi))


def _color_from_pitch(mu_f0: float) -> np.ndarray:
    """F0 → Hue → RGB. Deterministic color per fingerprint."""
    f0 = np.clip(mu_f0, 80.0, 400.0)
    hue = (f0 - 80.0) / (400.0 - 80.0)  # 0..1
    h = hue * 6.0
    c = 1.0
    x = c * (1.0 - abs(h % 2.0 - 1.0))
    if 0.0 <= h < 1.0:
        r, g, b = c, x, 0.0
    elif 1.0 <= h < 2.0:
        r, g, b = x, c, 0.0
    elif 2.0 <= h < 3.0:
        r, g, b = 0.0, c, x
    elif 3.0 <= h < 4.0:
        r, g, b = 0.0, x, c
    elif 4.0 <= h < 5.0:
        r, g, b = x, 0.0, c
    else:
        r, g, b = c, 0.0, x
    return np.array([r, g, b], dtype=np.float32)


