"""Wrapper module to make core event dataclasses importable at the package root."""
from __future__ import annotations

from .src.core.events import (  # type: ignore
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
