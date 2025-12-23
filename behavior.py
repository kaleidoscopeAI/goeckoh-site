"""Lightweight behavior monitoring and strategy hints (no heavy deps)."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List


@dataclass
class BehaviorMonitor:
    """
    Tracks repeated corrections, perseveration, and high-energy speech.
    Pure Python: safe to ship in a minimal build.
    """

    max_phrase_history: int = 5
    anxious_threshold: int = 3
    perseveration_threshold: int = 3
    high_energy_rms: float = 0.08

    def __post_init__(self) -> None:
        self._phrases: Deque[str] = deque(maxlen=self.max_phrase_history)
        self._correction_streak = 0
        self._last_event: str | None = None

    def register(self, normalized_text: str, needs_correction: bool, rms: float) -> str | None:
        """
        Update state and return an event string like "anxious", "perseveration", or "high_energy".
        """
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


@dataclass(frozen=True, slots=True)
class Strategy:
    category: str
    title: str
    description: str


STRATEGIES: List[Strategy] = [
    Strategy(
        category="mindfulness",
        title="Breathe + Ground",
        description="Guide a short breathing or grounding game to lower anxiety spikes.",
    ),
    Strategy(
        category="sensory",
        title="Sensory Break",
        description="Offer headphones, a fidget, or a brief quiet break if energy is high.",
    ),
    Strategy(
        category="encouragement",
        title="Positive Reinforcement",
        description="Acknowledge effort and invite a brief pause if corrections stack up.",
    ),
    Strategy(
        category="focus",
        title="Refocus Prompt",
        description="Gently redirect with a clear, short prompt if perseveration is detected.",
    ),
]

EVENT_TO_STRATEGIES: Dict[str, List[str]] = {
    "anxious": ["mindfulness", "sensory"],
    "high_energy": ["sensory"],
    "perseveration": ["focus"],
    "encouragement": ["encouragement"],
}


class StrategyAdvisor:
    """Returns a small list of strategy hints for a given event."""

    def suggest(self, event: str) -> List[Strategy]:
        cats = EVENT_TO_STRATEGIES.get(event)
        if not cats:
            return STRATEGIES[:1]
        ordered: List[Strategy] = []
        for cat in cats:
            ordered.extend([s for s in STRATEGIES if s.category == cat])
        return ordered or STRATEGIES[:1]
