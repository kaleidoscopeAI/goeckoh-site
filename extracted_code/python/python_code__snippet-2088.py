def __init__(self):
    self.use_neural = COQUI_AVAILABLE
    if self.use_neural:
        # Initialize Coqui (This downloads models on first run)
        # Using a lightweight model for reasonable latency
        print("Loading Neural Voice Model...")
        self.model = TTS("tts_models/multilingual/multi-dataset/your_tts", gpu=False)
    else:
        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', 150) 

def generate_speech_pcm(self, text: str, clone_ref_wav: str = None) -> np.ndarray:
    """
    Generates raw float32 PCM data from text.
    """
    try:
        if self.use_neural and clone_ref_wav:
            # High Fidelity Clone
            wav = self.model.tts(text=text, speaker_wav=clone_ref_wav, language="en")
            return np.array(wav, dtype=np.float32)

        elif not self.use_neural:
            # Standard System TTS Fallback
            # Capturing pyttsx3 output to buffer is tricky; usually plays directly.
            # For the exocortex 'Loop', we prioritize the Loop Logic over audio fidelity here.
            # In a full deploy, we'd write to temp file then read as numpy.
            return None 
    except Exception as e:
        print(f"[TTS Error] {e}")
        return None

