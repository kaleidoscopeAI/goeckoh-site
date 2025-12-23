tts: VoiceMimic
audio: AudioSettings
profile: VoiceProfile
config: VoiceCrystalConfig = field(default_factory=VoiceCrystalConfig)

def _smooth(self, wav: np.ndarray, window_ms: float) -> np.ndarray:
    if wav.size == 0:
        return wav
    window = max(int(window_ms * self.audio.sample_rate / 1000.0), 1)
    if window <= 1:
        return wav
    kernel = np.ones(window, dtype=np.float32) / float(window)
    return np.convolve(wav, kernel, mode="same").astype(np.float32)

def _apply_mode(self, wav: np.ndarray, mode: Mode) -> np.ndarray:
    if wav.size == 0:
        return wav
    if mode == "inner":
        smoothed = self._smooth(wav, self.config.inner_lowpass_window_ms * 2.0)
        rms = float(np.sqrt(np.mean(smoothed**2) + 1e-6))
        target_rms = rms * 0.8
        if rms > 0:
            smoothed = np.tanh(smoothed / rms) * target_rms
        return (smoothed * self.config.inner_volume_scale).astype(np.float32)
    if mode == "coach":
        return (wav * self.config.coach_volume_scale).astype(np.float32)
    return wav

def _synthesize_raw(self, text: str, style: Style) -> np.ndarray:
    if not text:
        return np.zeros(0, dtype=np.float32)
    sample = self.profile.pick_reference(style)
    if sample:
        self.tts.update_voiceprint(sample.path)
    return self.tts.synthesize(text)

def _apply_prosody(
    self,
    wav: np.ndarray,
    prosody_source_wav: Optional[np.ndarray],
    prosody_source_sr: Optional[int],
) -> np.ndarray:
    if prosody_source_wav is None or prosody_source_sr is None:
        return wav
    try:
        prosody = extract_prosody(prosody_source_wav, prosody_source_sr)
        return apply_prosody_to_tts(
            wav,
            self.audio.sample_rate,
            prosody,
            strength_pitch=self.config.prosody_strength_pitch,
            strength_energy=self.config.prosody_strength_energy,
        )
    except Exception:
        return wav

def speak(
    self,
    text: str,
    style: Style = "neutral",
    mode: Mode = "outer",
    prosody_source_wav: Optional[np.ndarray] = None,
    prosody_source_sr: Optional[int] = None,
) -> np.ndarray:
    base = self._synthesize_raw(text, style)
    if base.size == 0:
        return base
    prosody_applied = self._apply_prosody(base, prosody_source_wav, prosody_source_sr)
    processed = self._apply_mode(prosody_applied, mode)
    return processed

def say_inner(
    self,
    text: str,
    style: Style = "calm",
    prosody_source_wav: Optional[np.ndarray] = None,
    prosody_source_sr: Optional[int] = None,
) -> np.ndarray:
    return self.speak(
        text=text,
        style=style,
        mode="inner",
        prosody_source_wav=prosody_source_wav,
        prosody_source_sr=prosody_source_sr,
    )

def say_outer(
    self,
    text: str,
    style: Style = "neutral",
    prosody_source_wav: Optional[np.ndarray] = None,
    prosody_source_sr: Optional[int] = None,
) -> np.ndarray:
    return self.speak(
        text=text,
        style=style,
        mode="outer",
        prosody_source_wav=prosody_source_wav,
        prosody_source_sr=prosody_source_sr,
    )

def say_coach(
    self,
    text: str,
    style: Style = "excited",
    prosody_source_wav: Optional[np.ndarray] = None,
    prosody_source_sr: Optional[int] = None,
) -> np.ndarray:
    return self.speak(
        text=text,
        style=style,
        mode="coach",
        prosody_source_wav=prosody_source_wav,
        prosody_source_sr=prosody_source_sr,
    )
