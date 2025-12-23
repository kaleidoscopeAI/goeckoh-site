"""
Unified interface:
  synthesize(text: str, ref_wav: Optional[str], sample_rate: int) -> np.ndarray
"""

def __init__(self, tts_model_name: str | None = None, ref_wav: str | None = None, sample_rate: int = 16000):
    self.sample_rate = sample_rate
    self.available = VOICE_MIMIC_AVAILABLE
    self._engine = None
    if not VOICE_MIMIC_AVAILABLE:
        return

    settings = SpeechSettings(
        whisper_model="",
        language_tool_server=None,
        normalization_locale="en_US",
        tts_model_name=tts_model_name or "tts_models/multilingual/multi-dataset/xtts_v2",
        tts_voice_clone_reference=Path(ref_wav) if ref_wav else None,
        tts_sample_rate=sample_rate,
    )
    try:
        self._engine = VoiceMimic(settings)
    except Exception:
        self._engine = None
        self.available = False

def synthesize(self, text: str, ref_wav: Optional[str]) -> Optional[np.ndarray]:
    if not self._engine or not self.available or not text:
        return None
    # If a new ref wav is provided, update voiceprint
    if ref_wav:
        try:
            self._engine.update_voiceprint(Path(ref_wav))
        except Exception:
            pass
    try:
        audio = self._engine.synthesize(text)
        if audio is None:
            return None
        return np.asarray(audio, dtype=np.float32)
    except Exception:
        return None


