tts: VoiceMimic
profile: VoiceProfile
config: VoiceCrystalConfig

def _moving_average(self, wav: np.ndarray, window_ms: float) -> np.ndarray:
    window_size = int(self.config.sample_rate * (window_ms / 1000.0))\n        if window_size % 2 == 0:
        window_size += 1
    return np.convolve(wav, np.ones(window_size) / window_size, mode="same")

def _apply_mode(self, wav: np.ndarray, mode: Mode) -> np.ndarray:
    if mode == "inner":
        smoothed = self._moving_average(wav, self.config.inner_lowpass_window_ms)
        rms = np.sqrt(np.mean(smoothed**2) + 1e-6)
        target_rms = rms * self.config.inner_volume_scale
        smoothed = np.tanh(smoothed / (rms + 1e-6)) * target_rms
        return smoothed
    if mode == "coach":
        return wav * self.config.coach_volume_scale
    return wav

def speak(self, text: str, style: Style = "neutral", mode: Mode = "outer", prosody_source_wav: Optional[np.ndarray] = None, prosody_source_sr: Optional[int] = None) -> None:
    if not text:
        return

    ref = self.profile.pick_reference(style)
    if ref is not None:
        self.tts.update_voiceprint(ref)

    wav = self.tts.synthesize(text)
    if wav.size == 0:
        return

    if prosody_source_wav is not None:
        sr = prosody_source_sr or self.config.sample_rate
        pros = extract_prosody(prosody_source_wav, sr)
        wav = apply_prosody_to_tts(
            wav,
            self.config.sample_rate,
            pros,
            strength_pitch=self.config.prosody_strength_pitch,
            strength_energy=self.config.prosody_strength_energy,
        )

    processed = self._apply_mode(wav, mode)
    # Play or save as needed; assuming audio_io in loop handles play
    print(f"Speaking in {mode} mode: {text}")  # Placeholder for play
