def __init__(self, settings: SpeechModelSettings) -> None:
    self.model = TTS(settings.tts_model, gpu=False)
    self.voice_ref = settings.tts_voice_clone_reference

def update_voiceprint(self, ref: Path) -> None:
    self.voice_ref = ref

def synthesize(self, text: str) -> np.ndarray:
    wav = self.model.tts(text, speaker_wav=self.voice_ref)
    return np.array(wav)
