"""Top-level shim to expose event dataclasses regardless of package context."""
from __future__ import annotations

from project.events import (  # type: ignore
    EchoEvent,
    HeartMetrics,
    BrainMetrics,
    AvatarFrame,
    CombinedSnapshot,
    now_ts,
)

__all__ = [
    "EchoEvent",
    "HeartMetrics",
    "BrainMetrics",
    "AvatarFrame",
    "CombinedSnapshot",
    "now_ts",
]
