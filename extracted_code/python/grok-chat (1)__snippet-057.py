from .grammar_correction import GrammarCorrector

class SpeechLoop:
    # ...

    def __post_init__(self) -> None:
        # ...
        self.corrector = GrammarCorrector()

        self.voice_crystal = VoiceCrystal(
            engine=pyttsx3.init(),
            profile=self.voice_profile,
            config=VoiceCrystalConfig(sample_rate=self.config.audio.sample_rate),
        )

    async def handle_chunk(self, chunk: np.ndarray) -> None:
        # ... (STT for raw)

        corrected = self.corrector.correct(raw)

        # ... (echo with corrected)
