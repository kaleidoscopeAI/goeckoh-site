import numpy as np
try:
    from TTS.api import TTS
    COQUI_AVAILABLE = True
except ImportError:
    COQUI_AVAILABLE = False

class VoiceEngine:
    def __init__(self):
        self.use_neural = COQUI_AVAILABLE
        if self.use_neural:
            print("Loading AI Voice Model...")
            # Using 'your_tts' for zero-shot cloning support
            self.model = TTS("tts_models/multilingual/multi-dataset/your_tts", gpu=False)

    def generate_speech_pcm(self, text: str, clone_ref_wav: str = None) -> np.ndarray:
        try:
            if self.use_neural and clone_ref_wav:
                wav = self.model.tts(text=text, speaker_wav=clone_ref_wav, language="en")
                return np.array(wav, dtype=np.float32)
            return None
        except Exception as e:
            print(f"[TTS Gen Error] {e}")
            return None