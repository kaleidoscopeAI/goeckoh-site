import sounddevice as sd
import numpy as np

class AudioIO:
    def __init__(self, audio_cfg):
        self.cfg = audio_cfg

    def record_phrase(self, seconds: float) -> np.ndarray:
        print(f"Recording {seconds}s...")
        audio = sd.rec(int(seconds * self.cfg.sample_rate),
                       samplerate=self.cfg.sample_rate, channels=1, dtype='float32')
        sd.wait()
        return audio.flatten()

    def play_audio(self, audio: np.ndarray):
        sd.play(audio, samplerate=self.cfg.sample_rate)
        sd.wait()

    def rms(self, chunk): 
        return np.sqrt(np.mean(chunk**2))

