"""Alias around SystemSettings to avoid refactoring every import immediately."""

heart: HeartSettings = field(default_factory=HeartSettings)

def __post_init__(self) -> None:
    # ensure directories exist
    self.paths.ensure()


