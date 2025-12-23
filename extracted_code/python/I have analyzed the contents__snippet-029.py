def __init__(self):
    # Try to load Coqui TTS, fallback to pyttsx3 if missing/slow
    try:
        from TTS.api import TTS
        # Using a smaller, faster model for "production ready" responsiveness
        self.tts = TTS("tts_models/en/ljspeech/glow-tts", progress_bar=False, gpu=torch.cuda.is_available())
        self.engine_type = "neural"
        print("[VoiceCrystal] Neural TTS loaded.")
    except Exception as e:
        print(f"[VoiceCrystal] Neural TTS failed ({e}). Using standard TTS.")
        import pyttsx3
        self.engine = pyttsx3.init()
        self.engine_type = "standard"

def speak(self, text: str, style: str = "neutral"):
    print(f"🗣️ [Echo ({style})]: {text}")
    if self.engine_type == "neural":
        # In a real scenario, we would apply style transfer here
        # For speed, we simply synthesize to a temp file and play
        self.tts.tts_to_file(text=text, file_path="output.wav")
        data, fs = sf.read("output.wav")
        sd.play(data, fs)
        sd.wait()
    else:
        self.engine.say(text)
        self.engine.runAndWait()

