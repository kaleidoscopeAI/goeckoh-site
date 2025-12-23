class VoiceCrystal:
    def __init__(self, config):
        self.config = config
        self.voice_dir = config.base_dir / "voices"
        self.voice_dir.mkdir(exist_ok=True)
        self.fallback_engine = None
        if not XTTS_AVAILABLE:
            try:
                self.fallback_engine = pyttsx3.init()
                self.fallback_engine.setProperty('rate', 140)
                self.fallback_engine.setProperty('pitch', 200)  # child-like
            except Exception as e:
                logging.critical(f"pyttsx3 init failed: {e}")

    def say(self, text: str, style: Style = "neutral", prosody_source: np.ndarray | None = None):
        if not text.strip():
            return

        try:
            samples = list(self.voice_dir.glob("*.wav"))
            speaker_wav = str(samples[0]) if samples else None

            if XTTS_AVAILABLE:
                temp_path = self.config.base_dir / "temp_echo.wav"
                TTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=False, progress_bar=False).tts_to_file(
                    text=text, speaker_wav=speaker_wav, language="en", file_path=str(temp_path)
                )
                wav, sr = sf.read(temp_path)
                temp_path.unlink(missing_ok=True)
            else:
                if not self.fallback_engine:
                    logging.error("No TTS available")
                    return
                temp_path = self.config.base_dir / "fallback.wav"
                self.fallback_engine.save_to_file(text, str(temp_path))
                self.fallback_engine.runAndWait()
                wav, sr = sf.read(temp_path)
                temp_path.unlink(missing_ok=True)

            # Prosody transfer even on fallback
            if prosody_source is not None and len(prosody_source) > 1600:
                try:
                    f0, _, _ = librosa.pyin(prosody_source, fmin=75, fmax=600)
                    f0_mean = np.nanmean(f0[np.isfinite(f0)]) if np.any(np.isfinite(f0)) else 180
                    wav = librosa.effects.pitch_shift(wav, sr=sr, n_steps=np.log2(f0_mean / 180) * 12)
                    energy_src = np.sqrt(np.mean(prosody_source ** 2))
                    energy_tgt = np.sqrt(np.mean(wav ** 2))
                    if energy_tgt > 0:
                        wav *= energy_src / energy_tgt
                except Exception as e:
                    logging.warning(f"Prosody transfer failed: {e}")

            sd.play(wav, samplerate=sr)
            sd.wait()
        except Exception as e:
            logging.error(f"Voice synthesis failed: {e}\n{traceback.format_exc()}")
            # Ultimate fallback — do nothing silently if everything fails

    def harvest_facet(self, audio: np.ndarray, style: Style = "neutral"):
        try:
            timestamp = int(time.time())
            path = self.voice_dir / f"facet_{timestamp}_{style}.wav"
            sf.write(path, audio, self.config.sample_rate)
            logging.info(f"Harvested new voice facet: {path.name}")
        except Exception as e:
            logging.error(f"Failed to harvest facet: {e}")
