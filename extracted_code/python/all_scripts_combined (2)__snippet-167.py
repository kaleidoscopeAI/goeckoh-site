"""
Text-to-Speech engine using Coqui TTS.
"""

def __init__(self, settings: SpeechSettings):
    self.settings = settings
    self.device = "cuda" if torch.cuda.is_available() else "cpu"
    self.model = TTS(self.settings.tts_model_name).to(self.device)
    self.voice_sample = self.settings.tts_voice_clone_reference

def update_voiceprint(self, voice_sample: Path):
    self.voice_sample = voice_sample

def synthesize(self, text: str) -> np.ndarray:
    """
    Synthesizes text and returns the audio as a numpy array.
    """
    if not self.voice_sample or not self.voice_sample.exists():
        return self._fallback_synthesize(text)

    return np.array(
        self.model.tts(
            text=text,
            speaker_wav=str(self.voice_sample),
            language="en",
        )
    )

def _fallback_synthesize(self, text: str) -> np.ndarray:
    """
    Fallback to a default voice if no voice sample is available.
    """
    return np.array(self.model.tts(text=text, speaker=self.model.speakers[0], language="en"))
