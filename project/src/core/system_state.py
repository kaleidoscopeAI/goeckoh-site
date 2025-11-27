# ECHO_V4_UNIFIED/system_state.py
from __future__ import annotations
from dataclasses import dataclass
from threading import RLock
from typing import Optional

# These imports will be resolved from the other files we've created
from events import EchoEvent, HeartMetrics, BrainMetrics, AvatarFrame, now_ts

@dataclass
class SystemSnapshot:
    """A single, atomic snapshot of the entire system's state at a moment in time."""
    timestamp: float
    last_echo: Optional[EchoEvent]
    heart: Optional[HeartMetrics]
    brain: Optional[BrainMetrics]
    caption: str
    avatar: Optional[AvatarFrame]

class SystemState:
    """
    A thread-safe, in-memory object that serves as the single source of truth
    for the entire application. The backend (SpeechLoop) writes to it, and the
    frontend (GUI) reads from it.
    """
    def __init__(self) -> None:
        self._lock = RLock()
        self._snapshot = SystemSnapshot(
            timestamp=now_ts(),
            last_echo=None,
            heart=None,
            brain=None,
            caption="",
            avatar=None,
        )

    def update(
        self,
        echo: EchoEvent,
        heart: HeartMetrics,
        brain: BrainMetrics,
        caption: str,
        avatar: AvatarFrame,
    ) -> None:
        """Atomically updates the system snapshot with new data from the speech loop."""
        with self._lock:
            self._snapshot = SystemSnapshot(
                timestamp=now_ts(),
                last_echo=echo,
                heart=heart,
                brain=brain,
                caption=caption,
                avatar=avatar,
            )

    def get_snapshot(self) -> SystemSnapshot:
        """Returns a thread-safe copy of the current system snapshot for the GUI to render."""
        with self._lock:
            # The dataclass is immutable, so returning it is safe.
            return self._snapshot
