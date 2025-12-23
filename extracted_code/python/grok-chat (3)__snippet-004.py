class VoiceFragment:
    embedding: np.ndarray
    timestamp: float
    success_score: float  # 0–1 from phenotyping

class VoiceCrystal:
    def __init__(self):
        self.fragments: List[VoiceFragment] = []
        self.current_embedding = np.zeros(256, dtype=np.float32)
        self.lock = RLock()

    def add_fragment(self, audio: np.ndarray, success_score: float):
        with self.lock:
            y, sr = librosa.load(audio, sr=22050)
            emb = self._extract_embedding(y)
            self.fragments.append(VoiceFragment(emb, time.time(), success_score))
            # Equation 78 — Autopoietic Identity Maintenance
            alpha = 0.02 if success_score > 0.8 else 0.005
            self.current_embedding = (1 - alpha) * self.current_embedding + alpha * emb
            self._prune_old_fragments()

    def _extract_embedding(self, y):
        # Placeholder for actual speaker embedding (use ecapa-tdnn or titanet)
        return np.random.randn(256).astype(np.float32)  # REAL: replace with model

    def _prune_old_fragments(self):
        if len(self.fragments) > 500:
            self.fragments = sorted(self.fragments, key=lambda x: x.timestamp, reverse=True)[:400]

    def synthesize(self, text: str, style: str = "inner") -> np.ndarray:
        engine = pyttsx3.init()
        engine.setProperty('rate', 140 if style == "calm" else 170)
        engine.setProperty('volume', 0.7 if style == "inner" else 0.9)
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        engine.save_to_file(text, tmp.name)
        engine.runAndWait()
        y, sr = librosa.load(tmp.name, sr=22050)
        os.unlink(tmp.name)
        return y

