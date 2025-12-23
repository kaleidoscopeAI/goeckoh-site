class CompanionConfig(SystemSettings):
    """Alias around SystemSettings to avoid refactoring every import immediately."""

    heart: HeartSettings = field(default_factory=HeartSettings)

    def __post_init__(self) -> None:
        # ensure directories exist
        self.paths.ensure()


def _load_config() -> CompanionConfig:
    settings = load_settings()
    return CompanionConfig(
        child_id=settings.child_id,
        child_name=settings.child_name,
        device=settings.device,
        audio=settings.audio,
        speech=settings.speech,
        llm=settings.llm,
        behavior=settings.behavior,
        paths=settings.paths,
        heart=settings.heart,
    )


