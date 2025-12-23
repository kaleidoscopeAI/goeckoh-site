class VoiceCrystal:
    def __init__(self, config):
        self.config = config
        self.tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=False, progress_bar=False)
        self.samples: dict[Style, list[Path]] = {"calm": [], "neutral": [], "excited": []}
        self.load_samples()

    def load_samples(self):
        for style in self.samples:
            path = self.config.base_dir / "voices" / style
            path.mkdir(parents=True, exist_ok=True)
            self.samples[style] = list(path.glob("*.wav"))[:32]  # max 32 per style

    def say_in_child_voice(self, text: str, style: Style = "neutral", prosody_source: Optional[np.ndarray] = None) -> None:
        if not text.strip():
            return

        # Pick best sample for style
        candidates = self.samples.get(style) or self.samples["neutral"]
        speaker_wav = str(candidates[0]) if candidates else None

        temp_path = "temp_echo.wav"
        self.tts.tts_to_file(text=text, speaker_wav=speaker_wav, language="en", file_path=temp_path)

        wav, sr = sf.read(temp_path)

        if prosody_source is not None and len(prosody_source) > 1000:
            # Prosody transfer (F0 + energy)
            f0, _, _ = librosa.pyin(prosody_source, fmin=75, fmax=600)
            f0_mean = np.nanmean(f0) if not np.isnan(f0).all() else 200

            # Pitch shift to match child's natural pitch
            wav = librosa.effects.pitch_shift(wav, sr=sr, n_steps=np.log2(f0_mean/200)*12)

            # Energy matching
            energy_child = np.sqrt(np.mean(prosody_source**2))
            energy_tts = np.sqrt(np.mean(wav**2))
            if energy_tts > 0:
                wav = wav * (energy_child / energy_tts)

        # Play (headphone-only on mobile handled by platform)
        import sounddevice as sd
        sd.play(wav, samplerate=sr)
        sd.wait()
        Path(temp_path).unlink(missing_ok=True)

    def add_facet(self, audio: np.ndarray, style: Style):
        path = self.config.base_dir / "voices" / style / f"{len(self.samples[style])}.wav"
        sf.write(path, audio, 16000)
        self.samples[style].append(path)
