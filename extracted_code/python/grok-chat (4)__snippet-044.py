base_dir: Path = Path.home() / "Jacksons_Companion"
sample_rate: int = 16000
child_name: str = "Jackson"

def __post_init__(self):
    self.base_dir.mkdir(exist_ok=True)
    (self.base_dir / "voices").mkdir(exist_ok=True)
    (self.base_dir / "logs").mkdir(exist_ok=True)

