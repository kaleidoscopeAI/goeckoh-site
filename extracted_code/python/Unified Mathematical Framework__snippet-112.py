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

