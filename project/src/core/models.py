from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Literal, Optional


Timestamp = datetime


@dataclass(slots=True)
class AttemptRecord:
    """Structured representation of one speech attempt."""

    timestamp: Timestamp
    phrase_text: str
    raw_text: str
    corrected_text: str
    needs_correction: bool
    audio_file: Path
    similarity: float = 0.0


@dataclass(slots=True)
class BehaviorEvent:
    """Caregiver-facing behavior event that might trigger ABA scripts."""

    timestamp: Timestamp
    level: Literal["info", "warning", "critical"]
    category: Literal["anxious", "high_energy", "perseveration", "meltdown", "encouragement", "inner_echo", "caregiver_prompt"]
    title: str
    message: str
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass(slots=True)
class CaregiverPrompt:
    """Script submitted by Molly to be spoken gently in his own voice."""

    timestamp: Timestamp
    text: str
    mode: Literal["inner", "coach"] = "inner"
    active: bool = True
    expires_after_uses: int = 1


@dataclass(slots=True)
class VoiceFacet:
    """One voice sample tracked for passive adaptation."""

    path: Path
    style: Literal["neutral", "calm", "excited"]
    duration_s: float
    rms: float
    quality_score: float
    recorded_at: Timestamp

