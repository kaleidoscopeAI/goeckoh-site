class PathsConfig:
    base_dir: Path = Path.home() / "JacksonCompanion"
    voice_dir: Path = field(init=False)
    logs_dir: Path = field(init=False)
    routine_file: Path = field(init=False)
    speaker_ref_dir: Path = field(init=False)

    def __post_init__(self) -> None:
        self.base_dir.mkdir(exist_ok=True, parents=True)
        self.voice_dir = self.base_dir / "voice_crystal"
        self.logs_dir = self.base_dir / "logs"
        self.routine_file = self.base_dir / "routine.json"
        self.speaker_ref_dir = self.base_dir / "voice_samples"
        self.voice_dir.mkdir(exist_ok=True)
        self.logs_dir.mkdir(exist_ok=True)
        self.speaker_ref_dir.mkdir(exist_ok=True)


