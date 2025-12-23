"""
Coqui XTTS-based voice synthesis + prosody shaping.
"""

def __init__(self):
    self.tts = TTS(
        model_name="tts_models/multilingual/multi-dataset/xtts_v2",
        progress_bar=False,
        gpu=torch.cuda.is_available(),
    )
    self.samplerate = 16000
    self.current_pitch = 180.0
    self.lock = threading.Lock()
    self.ref_voices = self._load_reference_voices()

def _load_reference_voices(self) -> List[Path]:
    return sorted(VOICES_DIR.glob("*.wav"))

def add_fragment(self, audio: np.ndarray, success_score: float):
    """
    Learn prosody from a successful attempt.
    """
    with self.lock:
        y = audio.astype(np.float32).flatten()
        try:
            pitches, _ = librosa.piptrack(y=y, sr=self.samplerate)
            pitch_vals = pitches[pitches > 0]
            if pitch_vals.size > 0:
                pitch = float(np.mean(pitch_vals))
                self.current_pitch = 0.95 * self.current_pitch + 0.05 * pitch
        except Exception:
            pass

def synthesize(self, text: str, style: str = "inner") -> np.ndarray:
    text = enforce_first_person(text)
    with self.lock:
        ref_wavs = [str(p) for p in self.ref_voices] if self.ref_voices else None
        try:
            wav = self.tts.tts(
                text=text,
                speaker_wav=ref_wavs[0] if ref_wavs else None,
                language="en",
            )
        except Exception as e:
            print(f"[VoiceCrystal] TTS error: {e}")
            return np.zeros(1, dtype=np.float32)

    y = np.array(wav, dtype=np.float32)
    if style in ("calm", "inner"):
        b, a = butter(4, 800 / (self.samplerate / 2), btype="low")
        y = lfilter(b, a, y) * 0.6
    elif style == "excited":
        y = y * 1.1

    try:
        target_f0 = self.current_pitch or 180.0
        n_steps = np.log2(target_f0 / 180.0) * 12.0
        y = librosa.effects.pitch_shift(y, sr=self.samplerate, n_steps=n_steps)
    except Exception:
        pass

    return y.astype(np.float32)


