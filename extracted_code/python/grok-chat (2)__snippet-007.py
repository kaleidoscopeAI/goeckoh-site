import sounddevice as sd
import soundfile as sf
from pathlib import Path
from .config import AudioSettings

class AudioIO:
    def __init__(self, settings: AudioSettings):
        self.settings = settings

    def microphone_stream(self):
        with sd.InputStream(
            samplerate=self.settings.sample_rate,
            channels=1,
            dtype='float32',
            blocksize=int(self.settings.sample_rate * self.settings.chunk_seconds)
        ) as stream:
            while True:
                chunk, overflow = stream.read(stream.blocksize)
                if overflow:
                    print(\"Audio overflow detected.\")
                yield chunk.flatten()

    def record_phrase(self, duration_s: float) -> np.ndarray:
        print(\"Recording... Speak now.\")
        return sd.rec(
            int(self.settings.sample_rate * duration_s),
            samplerate=self.settings.sample_rate,
            channels=1,
            dtype='float32'
        )

    def play(self, wav: np.ndarray) -> None:
        sd.play(wav, samplerate=self.settings.sample_rate)

    def rms(self, chunk: np.ndarray) -> float:
        return float(np.sqrt(np.mean(chunk**2)))

    def save_wav(self, wav: np.ndarray, path: Path) -> None:
        sf.write(path, wav, self.settings.sample_rate)
