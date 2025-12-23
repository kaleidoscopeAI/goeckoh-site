"""Simple heuristics that monitor behavior cues to trigger guidance."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque


@dataclass
class BehaviorMonitor:
    """Tracks repeated corrections, perseveration, and high-energy speech."""

    max_phrase_history: int = 5
    anxious_threshold: int = 3
    perseveration_threshold: int = 3
    high_energy_rms: float = 0.08

    def __post_init__(self) -> None:
        self._phrases: Deque[str] = deque(maxlen=self.max_phrase_history)
        self._correction_streak = 0
        self._last_event: str | None = None

    def register(self, normalized_text: str, needs_correction: bool, rms: float) -> str | None:
        event: str | None = None

        if needs_correction:
            self._correction_streak += 1
        else:
            if self._correction_streak >= self.anxious_threshold:
                event = "encouragement"
            self._correction_streak = 0

        self._phrases.append(normalized_text)

        if self._correction_streak >= self.anxious_threshold:
            event = "anxious"
        elif normalized_text and list(self._phrases).count(normalized_text) >= self.perseveration_threshold:
            event = event or "perseveration"
        elif rms >= self.high_energy_rms:
            event = event or "high_energy"

        if event == self._last_event and event not in {"perseveration", "encouragement"}:
            return None
        if event:
            self._last_event = event
        return event
