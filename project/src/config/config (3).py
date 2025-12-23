"""Backward compatibility config shim."""
from __future__ import annotations

from dataclasses import dataclass, field

from core.paths import PathRegistry as Paths
from core.settings import (
    AudioSettings,
    BehaviorSettings,
    HeartSettings,
    SpeechSettings as SpeechModelSettings,
    SystemSettings,
    load_settings,
)


# Provide original name for downstream modules
EchoHeartConfig = HeartSettings


@dataclass(slots=True)
class CompanionConfig(SystemSettings):
    """Alias around SystemSettings to avoid refactoring every import immediately."""

    heart: HeartSettings = field(default_factory=HeartSettings)

    def __post_init__(self) -> None:
        # ensure directories exist
        self.paths.ensure()


def _load_config() -> CompanionConfig:
    settings = load_settings()
    return CompanionConfig(
        child_id=settings.child_id,
        child_name=settings.child_name,
        device=settings.device,
        audio=settings.audio,
        speech=settings.speech,
        llm=settings.llm,
        behavior=settings.behavior,
        paths=settings.paths,
        heart=settings.heart,
    )


CONFIG = _load_config()