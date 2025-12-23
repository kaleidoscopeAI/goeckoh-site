"""
A low-level wrapper for the Coqui XTTS voice cloning and synthesis engine.
"""
def __init__(self, config: SpeechModelSettings, device: str = "cpu"):
    self.config = config
    self.device = device
    self.tts: Optional[TTS] = None
    self.current_ref_path: Optional[str] = None

    if not _HAS_TTS:
        raise RuntimeError("TTS library is not installed. Voice synthesis is disabled.")

    print(f"[VoiceMimic] Loading TTS model: {config.tts_model_name} on {self.device}...")
    try:
        self.tts = TTS(model_name=config.tts_model_name, progress_bar=False).to(self.device)
        if config.tts_voice_clone_reference and config.tts_voice_clone_reference.exists():
            self.current_ref_path = str(config.tts_voice_clone_reference)
            print(f"[VoiceMimic] Using default voice reference: {self.current_ref_path}")
    except Exception as e:
        raise RuntimeError(f"Failed to load Coqui TTS model. Error: {e}")

def update_voiceprint(self, wav_path: Path) -> bool:
    """
    Updates the reference audio file for voice cloning.
    Returns True if the path is valid, False otherwise.
    """
    if wav_path.exists() and wav_path.is_file():
        self.current_ref_path = str(wav_path)
        return True
    print(f"[VoiceMimic] Warning: Voice reference path not found: {wav_path}")
    return False

def synthesize(self, text: str) -> np.ndarray:
    """
    Synthesizes audio from text using the current voiceprint.
    """
    if not self.tts:
        print("[VoiceMimic] Error: TTS model not loaded.")
        return np.array([], dtype=np.float32)

    if not text:
        return np.array([], dtype=np.float32)

    if not self.current_ref_path:
        print("[VoiceMimic] Warning: No voice reference set. Using default speaker.")
        # Fallback to default TTS without cloning
        wav = self.tts.tts(text=text, speaker=self.tts.speakers[0], language=self.tts.languages[0])
    else:
        try:
            # Use XTTS voice cloning
            wav = self.tts.tts(
                text=text,
                speaker_wav=self.current_ref_path,
                language="en" # XTTS requires a language hint
            )
        except Exception as e:
            print(f"[VoiceMimic] TTS synthesis failed: {e}. Falling back to default speaker.")
            # Fallback on error
            wav = self.tts.tts(text=text, speaker=self.tts.speakers[0], language=self.tts.languages[0])

    return np.array(wav, dtype=np.float32)
