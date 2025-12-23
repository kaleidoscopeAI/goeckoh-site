"""
Prosody-aware voice mimic.

- Learns average pitch from Jackson's real fragments.
- Feeds RMS into crystal.
- Uses XTTS (if available) with speaker reference clips in paths.speaker_ref_dir.
  Otherwise falls back to pyttsx3 + pitch shift.
"""

def __init__(self, paths: PathsConfig, crystal: ConsciousCrystalSystem):
    self.paths = paths
    self.crystal = crystal
    self.current_pitch: float = 180.0
    self.current_rate: int = 150
    self.engine = pyttsx3.init()
    self.engine.setProperty("rate", self.current_rate)
    self.lock = threading.Lock()
    self.tts: Optional[TTS] = None
    self.speaker_embedding: Optional[np.ndarray] = None

    if HAS_TTS:
        try:
            self.tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2")
            print("[VOICE] XTTS loaded.")
        except Exception as e:
            print(f"[VOICE] XTTS load failed, using pyttsx3 fallback: {e}")
            self.tts = None

    if self.tts is not None:
        self._load_or_build_speaker_embedding()

def _load_or_build_speaker_embedding(self) -> None:
    emb_path = self.paths.voice_dir / "speaker_embedding.npy"
    if emb_path.exists():
        self.speaker_embedding = np.load(emb_path)
        print("[VOICE] Loaded existing speaker embedding.")
        return

    wavs = list(self.paths.speaker_ref_dir.glob("*.wav"))
    if not wavs:
        print("[VOICE] No reference wavs in voice_samples; XTTS will use default speaker.")
        self.speaker_embedding = None
        return

    ref_wav = str(wavs[0])
    print(f"[VOICE] Building speaker embedding from {ref_wav}")
    try:
        self.speaker_embedding = self.tts.get_speaker_embedding(ref_wav)
        np.save(emb_path, self.speaker_embedding)
        print("[VOICE] Speaker embedding saved.")
    except Exception as e:
        print(f"[VOICE] Failed to build speaker embedding: {e}")
        self.speaker_embedding = None

def add_fragment(self, audio: np.ndarray, success_score: float) -> None:
    with self.lock:
        y = audio.astype(np.float32).flatten()
        if y.size == 0:
            return
        pitches, _ = librosa.piptrack(y=y, sr=CONFIG.audio.sample_rate)
        flat = pitches.flatten()
        voiced = flat[flat > 0]
        if voiced.size > 0:
            pitch = float(np.mean(voiced))
        else:
            pitch = 180.0
        self.current_pitch = 0.95 * self.current_pitch + 0.05 * pitch
        self.current_rate = 140 if success_score > 0.8 else 130
        self.engine.setProperty("rate", self.current_rate)

        rms = float(np.sqrt(np.mean(y**2)))
        self.crystal.update_from_rms(rms)

def synthesize(self, text: str, style: str = "inner") -> np.ndarray:
    """First-person inner voice / calm voice synthesis."""
    with self.lock:
        if self.tts is not None:
            try:
                out_path = self.paths.voice_dir / "tmp_xtts.wav"
                self.tts.tts_to_file(
                    text=text,
                    file_path=str(out_path),
                    speaker_wav=None if self.speaker_embedding is None else None,
                    language="en",
                )
                y, sr = librosa.load(str(out_path), sr=CONFIG.audio.sample_rate)
            except Exception as e:
                print(f"[VOICE] XTTS synthesis failed, falling back to pyttsx3: {e}")
                y = self._synthesize_pyttsx3(text)
                sr = CONFIG.audio.sample_rate
        else:
            y = self._synthesize_pyttsx3(text)
            sr = CONFIG.audio.sample_rate

        if style in ("calm", "inner"):
            b, a = butter(4, 800.0 / (sr / 2.0), btype="low")
            y = lfilter(b, a, y)
            y *= 0.6

        try:
            base_pitch = 180.0
            if self.current_pitch <= 0:
                self.current_pitch = base_pitch
            n_steps = float(math.log2(self.current_pitch / base_pitch) * 12.0)
            y = librosa.effects.pitch_shift(y, sr=sr, n_steps=n_steps)
        except Exception as e:
            print(f"[VOICE] Pitch shift error (continuing): {e}")

        return y.astype(np.float32)

def _synthesize_pyttsx3(self, text: str) -> np.ndarray:
    tmp = self.paths.voice_dir / "tmp_tts.wav"
    self.engine.save_to_file(text, str(tmp))
    self.engine.runAndWait()
    y, _ = librosa.load(str(tmp), sr=CONFIG.audio.sample_rate)
    try:
        os.remove(tmp)
    except Exception:
        pass
    return y


