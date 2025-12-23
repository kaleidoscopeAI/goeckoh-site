class SystemSettings:
    child_id: str = "child_001"
    child_name: str = "Jackson"
    device: str = "cpu"
    audio: AudioSettings = field(default_factory=AudioSettings)
    speech: SpeechSettings = field(default_factory=SpeechSettings)
    llm: LLMSettings = field(default_factory=LLMSettings)
    behavior: BehaviorSettings = field(default_factory=BehaviorSettings)
    paths: PathRegistry = field(default_factory=PathRegistry)
    heart: HeartSettings = field(default_factory=HeartSettings)

    @property
    def voice_sample(self) -> Path:
        return self.paths.voices_dir / "child_voice.wav"


def load_settings(config_path: Optional[Path] = None) -> SystemSettings:
    """
    Load settings from disk if available, otherwise fall back to defaults.
    """

    if config_path is None:
        config_path = DEFAULT_ROOT / "config.json"

    settings = SystemSettings()
    if config_path.exists():
        data = json.loads(config_path.read_text(encoding="utf-8"))
        _apply_json(settings, data)

    settings.paths.ensure_logs()
    return settings


def _apply_json(settings: SystemSettings, data: dict) -> None:
    for key, value in data.items():
        if key == "audio" and isinstance(value, dict):
            _update_dataclass(settings.audio, value)
        elif key == "speech" and isinstance(value, dict):
            _update_dataclass(settings.speech, value)
        elif key == "llm" and isinstance(value, dict):
            _update_dataclass(settings.llm, value)
        elif key == "behavior" and isinstance(value, dict):
            _update_dataclass(settings.behavior, value)
        elif key == "heart" and isinstance(value, dict):
            _update_dataclass(settings.heart, value)
        elif hasattr(settings, key):
            setattr(settings, key, value)


def _update_dataclass(obj, values: dict) -> None:
    for key, value in values.items():
        if hasattr(obj, key):
            setattr(obj, key, value)

