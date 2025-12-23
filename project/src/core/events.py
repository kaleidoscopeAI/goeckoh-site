# ECHO_V4_UNIFIED/events.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Any, List
import time
import numpy as np

Timestamp = float  # seconds since epoch

@dataclass(slots=True)
class EchoEvent:
    timestamp: Timestamp
    text_raw: str
    text_clean: str
    duration_s: float
    lang: str = "en"
    meta: Dict[str, Any] = field(default_factory=dict)

@dataclass(slots=True)
class HeartMetrics:
    timestamp: Timestamp
    stress: float
    harmony: float
    energy: float
    confidence: float
    temperature: float

@dataclass(slots=True)
class BrainMetrics:
    timestamp: Timestamp
    H_bits: float
    S_field: float
    L: float
    coherence: float
    phi: float

@dataclass(slots=True)
class AvatarFrame:
    timestamp: Timestamp
    positions: np.ndarray  # (N, 3)
    colors: np.ndarray   # (N, 3)
    sizes: np.ndarray    # (N,)
    meta: Dict[str, Any]

@dataclass(slots=True)
class CombinedSnapshot:
    timestamp: Timestamp
    last_echo_text: str
    heart: HeartMetrics
    brain: BrainMetrics
    caption: str
    avatar_meta: Dict[str, Any]

def now_ts() -> Timestamp:
    return time.time()
