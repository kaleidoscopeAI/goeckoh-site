"""Wrapper exposing SystemState/SystemSnapshot from src.core for top-level imports."""
from __future__ import annotations

from .src.core.system_state import SystemState, SystemSnapshot  # type: ignore

__all__ = ["SystemState", "SystemSnapshot"]
