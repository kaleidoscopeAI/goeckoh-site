class VoiceCrystal:
    def __init__(self, config):
        self.config = config
        self.tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=False, progress_bar=False)
        self.voice_dir = config.base_dir / "voices"
        self.voice_dir.mkdir(exist_ok=True)

    def say(self, text: str, style: Style = "neutral", prosody_source: np.ndarray | None = None):
        if not text.strip():
            return

        samples = list(self.voice_dir.glob("*.wav"))
        speaker_wav = str(samples[0]) if samples else None

        temp_path = self.config.base_dir / "temp_echo.wav"
        self.tts.tts_to_file(text=text, speaker_wav=speaker_wav, language="en", file_path=str(temp_path))

        wav, sr = sf.read(temp_path)

        if prosody_source is not None and len(prosody_source) > 1600:
            f0, _, _ = librosa.pyin(prosody_source, fmin=75, fmax=600)
            f0_mean = np.nanmean(f0[np.isfinite(f0)]) if np.any(np.isfinite(f0)) else 180

            wav = librosa.effects.pitch_shift(wav, sr=sr, n_steps=np.log2(f0_mean / 180) * 12)

            energy_src = np.sqrt(np.mean(prosody_source ** 2))
            energy_tgt = np.sqrt(np.mean(wav ** 2))
            if energy_tgt > 0:
                wav = wav * (energy_src / energy_tgt)

        sd.play(wav, samplerate=sr)
        sd.wait()
        temp_path.unlink(missing_ok=True)

    def harvest_facet(self, audio: np.ndarray, style: Style = "neutral"):
        timestamp = int(time.time())
        path = self.voice_dir / f"facet_{timestamp}_{style}.wav"
        sf.write(path, audio, self.config.sample_rate)

    def add_initial_facet(self, audio: np.ndarray):
        path = self.voice_dir / "initial_0.wav"
        sf.write(path, audio, self.config.sample_rate)

