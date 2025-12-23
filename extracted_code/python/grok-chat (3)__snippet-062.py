def __init__(self, heart_system, inner_heart):
    self.heart_system = heart_system
    self.inner_heart = inner_heart
    self.current_pitch = 180.0
    self.current_rate = 150
    self.engine = pyttsx3.init()
    self.engine.setProperty('rate', self.current_rate)
    self.lock = threading.Lock()

def add_fragment(self, audio: np.ndarray, success_score: float):
    with self.lock:
        y = audio.flatten()
        pitches, magnitudes = librosa.piptrack(y=y, sr=16000)
        pitch = np.mean([p for p in pitches.flatten() if p > 0]) if np.any(pitches > 0) else 180
        self.current_pitch = 0.95 * self.current_pitch + 0.05 * pitch
        self.current_rate = 140 if success_score > 0.8 else 130
        rms = np.sqrt(np.mean(y**2))
        inputs = torch.tensor([rms] * len(self.heart_system.graph), dtype=torch.float32)
        self.heart_system.update_energies(inputs)
        self.inner_heart.update_and_get_gcl(rms)

def synthesize(self, text: str, style: str = "inner") -> np.ndarray:
    with self.lock:
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        self.engine.save_to_file(text, tmp.name)
        self.engine.runAndWait()
        y, sr = librosa.load(tmp.name, sr=16000)
        os.unlink(tmp.name)
        if style in ["calm", "inner"]:
            b, a = butter(4, 800 / (sr / 2), btype='low')
            y = lfilter(b, a, y)
            y *= 0.6
        y = librosa.effects.pitch_shift(y, sr=sr, n_steps=np.log2(self.current_pitch / 180) * 12)
        return y.astype(np.float32)

