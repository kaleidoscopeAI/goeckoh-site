base_dir: Path = Path.home() / "Jacksons_Companion_Data"
sample_rate: int = 16000
min_utterance_seconds: float = 0.4
silence_threshold: float = 0.01
silence_duration_seconds: float = 1.5

def __post_init__(self):
    self.base_dir.mkdir(exist_ok=True)
    (self.base_dir / "voices").mkdir(exist_ok=True)
    (self.base_dir / "logs").mkdir(exist_ok=True)

