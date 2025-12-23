base_dir: Path = Path.home() / "Jacksons_Companion_Data"
sample_rate: int = 16000
min_utterance_seconds: float = 0.35
silence_threshold: float = 0.012
silence_duration_seconds: float = 1.4
max_buffer_seconds: int = 15

def __post_init__(self):
    self.base_dir.mkdir(parents=True, exist_ok=True)
    for sub in ["voices", "calming", "logs"]:
        (self.base_dir / sub).mkdir(exist_ok=True)

