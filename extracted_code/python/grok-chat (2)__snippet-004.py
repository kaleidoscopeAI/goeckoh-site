class Paths:
    base_dir: Path = Path.home() / "jacksons_companion"
    voices_dir: Path = field(init=False)
    logs_dir: Path = field(init=False)
    attempts_csv: Path = field(init=False)
    guidance_csv: Path = field(init=False)

    def __post_init__(self) -> None:
        self.voices_dir = self.base_dir / "voices"
        self.logs_dir = self.base_dir / "logs"
        self.attempts_csv = self.logs_dir / "attempts.csv"
        self.guidance_csv = self.logs_dir / "guidance_events.csv"
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.voices_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)

