class VoiceMimic:
    """Handles speech synthesis in the child's voice."""

    settings: SpeechModelSettings

    def __post_init__(self) -> None:
        self._engine = pyttsx3.init()
        self._voice_bands: Sequence[Tuple[int, int]] = (
            (80, 300),
            (300, 1200),
            (1200, 4000),
            (4000, 8000),
        )
        self._voiceprint_eq: Optional[np.ndarray] = None
        if self.settings.tts_model_name:
            self._maybe_set_voice(self.settings.tts_model_name)
        if self.settings.tts_voice_clone_reference and self.settings.tts_voice_clone_reference.exists():
            self.update_voiceprint(self.settings.tts_voice_clone_reference)

    def synthesize(self, text: str) -> np.ndarray:
        if not text:
            return np.zeros(0, dtype=np.float32)

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = Path(tmp.name)

        try:
            self._engine.save_to_file(text, str(tmp_path))
            self._engine.runAndWait()
            wav, sr = sf.read(tmp_path, dtype="float32")
        finally:
            tmp_path.unlink(missing_ok=True)

        wav = _ensure_mono(np.asarray(wav, dtype=np.float32))
        if sr != self.settings.tts_sample_rate:
            wav = librosa.resample(wav, orig_sr=sr, target_sr=self.settings.tts_sample_rate)
        if self._voiceprint_eq is not None:
            wav = self._apply_voiceprint(wav)
        return np.asarray(wav, dtype=np.float32)

    def update_voiceprint(self, new_reference: Path) -> None:
        if not new_reference.exists():
            return
        wav, sr = sf.read(new_reference, dtype="float32")
        wav = _ensure_mono(np.asarray(wav, dtype=np.float32))
        if sr != self.settings.tts_sample_rate:
            wav = librosa.resample(wav, orig_sr=sr, target_sr=self.settings.tts_sample_rate)
        self._voiceprint_eq = self._extract_eq_profile(wav)

    def _maybe_set_voice(self, desired: str) -> None:
        try:
            voices = self._engine.getProperty("voices") or []
        except Exception:
            return
        desired = desired.lower()
        for voice in voices:
            if desired in voice.id.lower() or desired in getattr(voice, "name", "").lower():
                self._engine.setProperty("voice", voice.id)
                return

    def _extract_eq_profile(self, wav: np.ndarray) -> Optional[np.ndarray]:
        if wav.size == 0:
            return None
        spectrum = np.abs(np.fft.rfft(wav))
        freqs = np.fft.rfftfreq(len(wav), 1.0 / self.settings.tts_sample_rate)
        eq_profile = []
        for low, high in self._voice_bands:
            mask = (freqs >= low) & (freqs < high)
            if not np.any(mask):
                eq_profile.append(1.0)
                continue
            eq_profile.append(float(spectrum[mask].mean() + 1e-8))
        eq = np.asarray(eq_profile, dtype=np.float32)
        avg = float(np.mean(eq) or 1.0)
        eq = np.clip(eq / avg, 0.5, 2.5)
        return eq

    def _apply_voiceprint(self, wav: np.ndarray) -> np.ndarray:
        if wav.size == 0 or self._voiceprint_eq is None:
            return wav
        spectrum = np.fft.rfft(wav)
        freqs = np.fft.rfftfreq(len(wav), 1.0 / self.settings.tts_sample_rate)
        shaped = spectrum.copy()
        for gain, (low, high) in zip(self._voiceprint_eq, self._voice_bands):
            mask = (freqs >= low) & (freqs < high)
            shaped[mask] *= gain
        result = np.fft.irfft(shaped, n=len(wav))
        max_val = np.max(np.abs(result)) or 1.0
        normalized = result / max(1.0, max_val)
        return np.asarray(normalized, dtype=np.float32)


