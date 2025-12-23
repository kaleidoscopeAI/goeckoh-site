class Paths:
    base: Path = Path.home() / "EchoCrystal"
    voices_dir: Path = base / "voices"
    logs_dir: Path = base / "logs"
    metrics_csv: Path = logs_dir / "aba_progress.csv"

    def __post_init__(self):
        self.base.mkdir(exist_ok=True)
        self.voices_dir.mkdir(exist_ok=True)
        self.logs_dir.mkdir(exist_ok=True)

