"""Central configuration for the autism speech companion."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(slots=True)
class Paths:
    """Structure holding all filesystem locations used by the system."""

    base_dir: Path = Path.home() / "speech_companion"
    voices_dir: Path | None = None
    logs_dir: Path | None = None
    metrics_csv: Path | None = None
    guidance_csv: Path | None = None

    def __post_init__(self) -> None:
        self.voices_dir = self.base_dir / "voices"
        self.logs_dir = self.base_dir / "logs"
        self.metrics_csv = self.logs_dir / "attempts.csv"
        self.guidance_csv = self.logs_dir / "guidance_events.csv"
        self.ensure()

    def ensure(self) -> None:
        """Create the directories if they are missing."""
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.voices_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)


@dataclass(slots=True)
class AudioSettings:
    """All audio constants in one place."""

    sample_rate: int = 16_000
    chunk_seconds: float = 2.5
    silence_rms_threshold: float = 0.0125
    channels: int = 1


@dataclass(slots=True)
class SpeechModelSettings:
    """Metadata about ML models used by the speech pipeline."""

    whisper_model: str = "base"
    language_tool_server: str | None = None  # use default JVM unless a remote server is provided
    tts_model_name: str = ""  # Optional pyttsx3 voice id/name hint
    tts_voice_clone_reference: Path | None = None
    tts_sample_rate: int = 16_000


@dataclass(slots=True)
class BehaviorSettings:
    """High level behavior knobs parents can toggle."""

    correction_echo_enabled: bool = True


@dataclass(slots=True)
class CompanionConfig:
    """Top-level configuration shared across CLI, GUI, and agent layers."""

    child_id: str = "child_001"
    child_name: str = "Companion User"
    caregiver_name: str = "Caregiver"
    paths: Paths = field(default_factory=Paths)
    audio: AudioSettings = field(default_factory=AudioSettings)
    speech: SpeechModelSettings = field(default_factory=SpeechModelSettings)
    behavior: BehaviorSettings = field(default_factory=BehaviorSettings)


CONFIG = CompanionConfig()
