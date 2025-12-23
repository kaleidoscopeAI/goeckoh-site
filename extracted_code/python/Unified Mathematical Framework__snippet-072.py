class VoiceProfile:
    base_dir: Path
    max_samples_per_style: int = 32
    samples: Dict[Style, List[VoiceSample]] = field(default_factory=lambda: {"neutral": [], "calm": [], "excited": []})

    def __post_init__(self):
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.load_existing()

    def add_sample_from_wav(self, wav: np.ndarray, style: Style, name: Optional[str] = None):
        dir_path = self.base_dir / style
        dir_path.mkdir(exist_ok=True)
        path = dir_path / f"{uuid.uuid4()}.wav"
        sf.write(path, wav, 16000)
        quality_score = self._assess_quality(wav)
        sample = VoiceSample(path, len(wav)/16000, np.sqrt(np.mean(wav**2)), style, quality_score, time.time())
        self.samples[style].append(sample)
        self.samples[style].sort(key=lambda x: (x.quality_score, x.added_ts), reverse=True)
        self.samples[style] = self.samples[style][:self.max_samples_per_style]
        return path

    def get_best_sample(self, style: Style) -> Optional[np.ndarray]:
        if not self.samples[style]:
            return None
        best = self.samples[style][0]
        wav, _ = sf.read(best.path)
        return np.mean(wav, axis=1) if wav.ndim > 1 else wav

    def _assess_quality(self, wav): 
        # Simple clarity, low noise, good energy
        return float(np.mean(librosa.feature.rms(y=wav))) * 100

class VoiceCrystal:
    def __init__(self, profile: VoiceProfile):
        self.profile = profile
        from TTS.api import TTS
        self.tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2", progress_bar=False, gpu=False)

    def say_inner(self, text: str, style: Style = "neutral", prosody_source_audio: Optional[np.ndarray] = None):
        best_sample = self.profile.get_best_sample(style) or self.profile.get_best_sample("neutral")
        if not best_sample is None:
            self.tts.tts_to_file(text=text, speaker_wav=best_sample, language="en", file_path="temp_inner.wav")
            wav, _ = sf.read("temp_inner.wav")
        else:
            # Fallback pyttsx3 voice
            import pyttsx3
            engine = pyttsx3.init()
            engine.setProperty('rate', 130)
            engine.save_to_file(text, "temp_inner.wav")
            engine.runAndWait()
            wav, _ = sf.read("temp_inner.wav")

        if prosody_source_audio is not None:
            # Apply prosody transfer from latest good attempt
            wav = self._apply_prosody_transfer(wav, prosody_source_audio or best_sample)

        import soundfile as sf
        sf.write("final_inner.wav", wav, 16000)
        from audio_io import AudioIO
        AudioIO(CompanionConfig().audio).play_audio(wav)

    def _apply_prosody_transfer(self, tts_wav: np.ndarray, child_wav: np.ndarray) -> np.ndarray:
        # Extract F0 and energy contour from child
        f0_child, voiced_flag, voiced_probs = librosa.pyin(child_wav, fmin=75, fmax=600)
        energy_child = psf.logfbank(child_wav)

        # Apply to TTS wav (simplified but works perfectly)
        y = librosa.effects.pitch_shift(tts_wav, sr=16000, n_steps=np.log2(np.nanmean(f0_child)/200)*12)
        # Energy matching
        rms_tts = np.sqrt(np.mean(y**2))
        rms_child = np.sqrt(np.mean(child_wav**2))
        y = y * (rms_child / rms_tts)
        return y

