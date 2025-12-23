"""
Lightweight "voice crystal".
- Tracks an evolving estimate of Jackson's average pitch.
- Uses pyttsx3 TTS then pitch-shifts toward his pitch.
- Applies optional lowpass + attenuation for "inner" / "calm" voice.
"""
def __init__(self):
    self.current_pitch = 180.0 # Hz, initial guess
    self.current_rate = 150
    self.engine = pyttsx3.init()
    self.engine.setProperty("rate", self.current_rate)
    self.lock = threading.Lock()
def add_fragment(self, audio: np.ndarray, success_score: float) -> None:
    """
    Update voice crystal stats from a new real utterance.
    """
    with self.lock:
        y = audio.astype(np.float32).flatten()
        if y.size == 0:
            return
        pitches, magnitudes = librosa.piptrack(y=y, sr=16000)
        flat_pitches = pitches.flatten()
        voiced = flat_pitches[flat_pitches > 0]
        if voiced.size > 0:
            pitch = float(np.mean(voiced))
        else:
            pitch = 180.0
        # Smooth pitch over time
        self.current_pitch = 0.95 * self.current_pitch + 0.05 * pitch
        # Slightly slower, calmer rate when success is high
        self.current_rate = 140 if success_score > 0.8 else 130
        self.engine.setProperty("rate", self.current_rate)
def synthesize(self, text: str, style: str = "inner") -> np.ndarray:
    """
    Synthesize text to audio, then approximate Jackson's pitch.
    """
    with self.lock:
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp_name = tmp.name
        tmp.close()
        # TTS → temp wav
        self.engine.save_to_file(text, tmp_name)
        self.engine.runAndWait()
        y, sr = librosa.load(tmp_name, sr=16000)
        os.unlink(tmp_name)
        # "Inner" / "calm" voice: low-pass filter, lower volume
        if style in ("calm", "inner"):
            b, a = butter(4, 800.0 / (sr / 2.0), btype="low")
            y = lfilter(b, a, y)
            y = y * 0.6
        # Approximate pitch shift toward current_pitch
        try:
            base_pitch = 180.0
            if self.current_pitch <= 0:
                self.current_pitch = base_pitch
            n_steps = float(np.log2(self.current_pitch / base_pitch) * 12.0)
            y = librosa.effects.pitch_shift(y, sr=sr, n_steps=n_steps)
        except Exception as e:
            print(f"[VOICE] Pitch shift error (continuing): {e}")
        return y.astype(np.float32)
