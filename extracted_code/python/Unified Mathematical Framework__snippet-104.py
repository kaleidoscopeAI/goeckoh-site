def __init__(self, config):
    self.config = config
    self.tts = TTS("tts_models/multilingual/multi-dataset/your_tts", progress_bar=False, gpu=False)
    self.reference_dir = Path(config.clone_reference_dir)
    self.reference_dir.mkdir(exist_ok=True)
    self.reference_files = list(self.reference_dir.glob("*.wav"))

def add_sample(self, wav: np.ndarray, style: str = "neutral"):
    path = self.reference_dir / f"{style}_{len(self.reference_files)}.wav"
    sf.write(path, wav, self.config.sample_rate)
    self.reference_files.append(path)

def synthesize(self, text: str, style: str = "neutral") -> np.ndarray:
    if not self.reference_files:
        # Graceful fallback
        import pyttsx3
        engine = pyttsx3.init()
        tmp = "fallback.wav"
        engine.save_to_file(text, tmp)
        engine.runAndWait()
        wav, _ = librosa.load(tmp, sr=self.config.sample_rate)
        os.remove(tmp)
        return wav

    # Use most recent or best sample
    ref_path = self.reference_files[-1]
    wav = self.tts.tts(text=text, speaker_wav=str(ref_path), language="en")
    return np.array(wav)

