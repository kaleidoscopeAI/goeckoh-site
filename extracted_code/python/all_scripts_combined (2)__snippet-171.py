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


