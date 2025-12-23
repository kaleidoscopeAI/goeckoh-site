"""
core package
=============

Shared configuration, data models, and logging utilities that every subsystem
uses.  Keeping these primitives in one place prevents circular imports between
the audio, speech, emotion, and ABA layers.
"""

from .paths import PathRegistry
from .settings import SystemSettings, load_settings
from .models import (
    AttemptRecord,
    BehaviorEvent,
    CaregiverPrompt,
    VoiceFacet,
)
from .core_logging import MetricsLogger, GuidanceLogger

__all__ = [
    "AttemptRecord",
    "BehaviorEvent",
    "CaregiverPrompt",
    "MetricsLogger",
    "GuidanceLogger",
    "PathRegistry",
    "SystemSettings",
    "VoiceFacet",
    "load_settings",
]
