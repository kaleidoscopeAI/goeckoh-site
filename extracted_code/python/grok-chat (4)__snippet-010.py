class Config:
    base_dir: Path = Path.home() / "Jacksons_Companion_Data"
    sample_rate: int = 16000
    child_name: str = "Jackson"
    vad_min_silence_ms: int = 1200      # Autism-tuned patience

    def __post_init__(self):
        self.base_dir.mkdir(exist_ok=True)
        (self.base_dir / "voices").mkdir(exist_ok=True)

