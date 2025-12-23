def __init__(self, config):
    self.config = config
    self.voice_dir = config.base_dir / "voices"
    self.voice_dir.mkdir(exist_ok=True)
    self.tts = None
    self.fallback = None
    try:
        self.tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=False)
    except:
        logging.warning("XTTS not available — using pyttsx3 fallback")
    if not self.tts:
        self.fallback = pyttsx3.init()
        self.fallback.setProperty('rate', 145)
        self.fallback.setProperty('volume', 1.0)

def say(self, text: str, style: Style = "neutral", prosody_source: np.ndarray | None = None, use_fallback: bool = False):
    if not text.strip():
        return

    try:
        samples = list(self.voice_dir.glob("*.wav"))
        speaker_wav = str(samples[0]) if samples else None

        temp = self.config.base_dir / "temp.wav"

        if self.tts and not use_fallback:
            self.tts.tts_to_file(text=text, speaker_wav=speaker_wav, language="en", file_path=str(temp))
        else:
            if self.fallback:
                self.fallback.save_to_file(text, str(temp))
                self.fallback.runAndWait()
            else:
                return  # silent fail if no voice at all

        wav, sr = sf.read(temp)

        if prosody_source is not None and len(prosody_source) > 1600:
            f0, _, _ = librosa.pyin(prosody_source, fmin=75, fmax=600)
            f0_mean = np.nanmean(f0[np.isfinite(f0)]) if np.any(np.isfinite(f0)) else 185
            wav = librosa.effects.pitch_shift(wav, sr=sr, n_steps=np.log2(f0_mean / 185) * 12)

            energy_src = np.sqrt(np.mean(prosody_source ** 2))
            energy_tgt = np.sqrt(np.mean(wav ** 2))
            if energy_tgt > 0:
                wav = wav * (energy_src / energy_tgt)

        sd.play(wav, samplerate=sr)
        sd.wait()
        temp.unlink(missing_ok=True)
    except Exception as e:
        logging.error(f"VoiceCrystal.say failed: {e}")

def harvest_facet(self, audio: np.ndarray, style: Style = "neutral"):
    try:
        path = self.voice_dir / f"facet_{int(time.time())}_{style}.wav"
        sf.write(path, audio, self.config.sample_rate)
        logging.info(f"Harvested facet saved: {path.name}")
    except Exception as e:
        logging.error(f"Harvest failed: {e}")

def add_initial_facet(self, audio: np.ndarray):
    path = self.voice_dir / "initial.wav"
    sf.write(path, audio, self.config.sample_rate)

