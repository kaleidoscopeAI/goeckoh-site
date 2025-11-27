from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal


DEFAULT_ROOT = Path.home() / "EchoSystem"


@dataclass(slots=True)
class PathRegistry:
    """
    Central place for every on-disk path the companion touches.

    The installer and runtime both derive from the same registry so assets end
    up in predictable places on Linux, macOS, Windows, and Raspberry Pi.
    """

    root: Path = DEFAULT_ROOT
    platform: Literal["linux", "darwin", "windows"] | None = None

    def __post_init__(self) -> None:
        self.platform = (self.platform or self._detect_platform()).lower()
        self.root.mkdir(parents=True, exist_ok=True)

    # --- Derived paths --------------------------------------------------
    @property
    def voices_dir(self) -> Path:
        return self.root / "voices"

    @property
    def attempts_dir(self) -> Path:
        return self.root / "logs" / "attempts"

    @property
    def logs_dir(self) -> Path:
        return self.root / "logs"

    @property
    def metrics_csv(self) -> Path:
        return self.logs_dir / "metrics.csv"

    @property
    def guidance_csv(self) -> Path:
        return self.logs_dir / "guidance_events.csv"

    @property
    def caregiver_prompts_csv(self) -> Path:
        return self.logs_dir / "caregiver_prompts.csv"

    @property
    def cache_dir(self) -> Path:
        return self.root / ".cache"

    @property
    def models_dir(self) -> Path:
        return self.root / "models"

    @property
    def dashboard_static_dir(self) -> Path:
        return self.root / "dashboard_static"

    @property
    def config_file(self) -> Path:
        return self.root / "config.json"

    # --- Helpers --------------------------------------------------------
    def ensure_logs(self) -> None:
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.attempts_dir.mkdir(parents=True, exist_ok=True)
        self.voices_dir.mkdir(parents=True, exist_ok=True)

    # Backwards compatibility with older modules expecting `Paths.ensure()`.
    def ensure(self) -> None:
        self.ensure_logs()

    def _detect_platform(self) -> str:
        from sys import platform

        if platform.startswith("linux"):
            return "linux"
        if platform == "darwin":
            return "darwin"
        return "windows"

