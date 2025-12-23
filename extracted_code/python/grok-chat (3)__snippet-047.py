def __init__(self):
    self.current_pitch = 180.0   # will adapt from child's fragments
    self.current_rate = 150
    self.engine = pyttsx3.init()
    self.engine.setProperty('rate', self.current_rate)
    self.lock = threading.Lock()

def add_fragment(self, audio: np.ndarray, success_score: float):
    with self.lock:
        # Extract average pitch from child's utterance
        y = audio.flatten()
        pitches, magnitudes = librosa.piptrack(y=y, sr=16000)
        pitch = np.mean([p for p in pitches.flatten() if p > 0]) if np.any(pitches > 0) else 180
        self.current_pitch = 0.95 * self.current_pitch + 0.05 * pitch
        self.current_rate = 140 if success_score > 0.8 else 130

def synthesize(self, text: str, style: str = "inner") -> np.ndarray:
    with self.lock:
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        self.engine.save_to_file(text, tmp.name)
        self.engine.runAndWait()
        y, sr = librosa.load(tmp.name, sr=16000)
        os.unlink(tmp.name)

        if style == "calm" or style == "inner":
            # Low-pass for inner voice feeling (like hearing your own head voice)
            b, a = butter(4, 800 / (sr / 2), btype='low')
            y = lfilter(b, a, y)
            y = y * 0.6  # quieter, gentle

        # Simple formant shift to make it feel like child's voice
        y = librosa.effects.pitch_shift(y, sr=16000, n_steps = (np.log2(self.current_pitch / 180) * 12))
        return y.astype(np.float32)

