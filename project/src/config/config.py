"""Unified configuration for Organic Seed + Echo speech companion.

This module intentionally serves two roles:

- ``Config`` / ``OrganicConfig``: original configuration used by the
  Organic AI seed + web crawler stack (main.py, seed.py).
- ``CompanionConfig`` / ``CONFIG``: high-level settings for the Echo
  speech companion, Crystalline Heart, and AGI substrate.  Most of the
  autism speech companion pipeline and dashboards depend on this.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List

from ..core.paths import PathRegistry, DEFAULT_ROOT
from .settings import (
    AudioSettings,
    SpeechSettings as SpeechModelSettings,
    LLMSettings,
    BehaviorSettings,
    HeartSettings,
    SystemSettings,
    load_settings,
)


# ---------------------------------------------------------------------------
# Organic Seed / web-crawler config (legacy)
# ---------------------------------------------------------------------------


@dataclass
class OrganicConfig:
    seed_name: str = "Seed.0"
    dna_size: int = 5
    max_energy: float = 10.0
    ethics_risk_threshold: float = 0.5
    replication_energy_cost: float = 4.0
    learning_cost_base: float = 0.6
    crawler_enabled: bool = True
    hid_enabled: bool = False
    ws_host: str = "127.0.0.1"
    ws_port: int = 8765
    anneal_steps: int = 128
    anneal_temp_start: float = 1.5
    anneal_temp_end: float = 0.05
    audio_sr: int = 22050
    audio_base_freq: float = 220.0
    allowed_domains: List[str] = field(default_factory=lambda: ["example.com"])


# Backwards-compat alias used by seed.py / organic_ai.py
Config = OrganicConfig


# ---------------------------------------------------------------------------
# Echo companion + Crystalline Heart / AGI config
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class DBConfig:
    """On-disk location for crystalline brain + snippet store."""

    db_path: Path = DEFAULT_ROOT / "echo_brain.sqlite3"


# Alias used by heart_core.py
EchoHeartConfig = HeartSettings


@dataclass(slots=True)
class CompanionConfig(SystemSettings):
    """
    Top-level configuration shared across CLI, GUI, speech loop,
    Crystalline Heart, and AGI substrate.
    """

    audio: AudioSettings = field(default_factory=AudioSettings)
    speech: SpeechModelSettings = field(default_factory=SpeechModelSettings)
    llm: LLMSettings = field(default_factory=LLMSettings)
    behavior: BehaviorSettings = field(default_factory=BehaviorSettings)
    paths: PathRegistry = field(default_factory=PathRegistry)
    heart: HeartSettings = field(default_factory=HeartSettings)
    db: DBConfig = field(default_factory=DBConfig)

    def __post_init__(self) -> None:
        # Ensure directories exist for logs, attempts, and voices.
        self.paths.ensure_logs()


def _load_companion_config() -> CompanionConfig:
    """Hydrate CompanionConfig from JSON on disk if present."""
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


# Global singleton used across the speech companion + AGI stack.
CONFIG: CompanionConfig = _load_companion_config()
