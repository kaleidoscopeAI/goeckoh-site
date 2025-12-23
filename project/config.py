"""Convenience wrapper to expose config objects from src/config for package-level imports."""
from __future__ import annotations

# Re-export the primary configuration classes and singleton
from .src.config.config import (  # type: ignore
    CompanionConfig,
    OrganicConfig,
    Config,
    CONFIG,
    DBConfig,
    EchoHeartConfig,
    SpeechModelSettings,
)
from .src.config.settings import (  # type: ignore
    AudioSettings,
    SpeechSettings,
    LLMSettings,
    BehaviorSettings,
    HeartSettings,
    SystemSettings,
)
from .src.core.paths import PathRegistry, DEFAULT_ROOT  # type: ignore

__all__ = [
    "CompanionConfig",
    "OrganicConfig",
    "Config",
    "CONFIG",
    "DBConfig",
    "EchoHeartConfig",
    "AudioSettings",
    "SpeechSettings",
    "SpeechModelSettings",
    "LLMSettings",
    "BehaviorSettings",
    "HeartSettings",
    "SystemSettings",
    "PathRegistry",
    "DEFAULT_ROOT",
]
