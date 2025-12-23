"""Neural voice cloning engine"""

def __init__(self, use_gpu=False):
    self.use_neural = False
    self.model = None
    if NEURAL_TTS_AVAILABLE:
        try:
            self.model = TTS("tts_models/en/vctk/vits", gpu=use_gpu)
            self.use_neural = True
        except Exception as e:
            print(f"TTS initialization failed: {e}")

def generate_speech_pcm(self, text: str, clone_ref_wav: str = None) -> Optional[np.ndarray]:
    """Generate speech with neural TTS"""
    if not self.use_neural or self.model is None:
        return None

    try:
        if clone_ref_wav and os.path.exists(clone_ref_wav):
            wav = self.model.tts(text=text, speaker_wav=clone_ref_wav, language="en")
            return np.array(wav, dtype=np.float32)
        else:
            wav = self.model.tts(text=text)
            return np.array(wav, dtype=np.float32)
    except Exception as e:
        print(f"Neural TTS generation failed: {e}")
        return None

