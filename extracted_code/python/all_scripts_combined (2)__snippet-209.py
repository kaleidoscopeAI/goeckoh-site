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


