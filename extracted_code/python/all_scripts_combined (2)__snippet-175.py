"""Wraps microphone capture and speaker playback."""

settings: AudioSettings

def __post_init__(self) -> None:
    self._q: "queue.Queue[np.ndarray]" = queue.Queue()

def microphone_stream(self) -> Generator[np.ndarray, None, None]:
    """Yield audio chunks from the default microphone."""

    def _callback(indata: np.ndarray, frames: int, time, status) -> None:  # type: ignore[override]
        if status:
            print(f"[AudioIO] stream status: {status}")
        self._q.put(indata.copy())

    with sd.InputStream(
        samplerate=self.settings.sample_rate,
        channels=self.settings.channels,
        dtype="float32",
        callback=_callback,
    ):
        while True:
            chunk = self._q.get()
            yield chunk

def rms(self, audio: np.ndarray) -> float:
    """Root mean square energy for silence detection."""
    if audio.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(audio))))

def record_phrase(self, seconds: float) -> np.ndarray:
    """Blocking recording helper used for canonical phrases."""
    frames = int(seconds * self.settings.sample_rate)
    recording = sd.rec(frames, samplerate=self.settings.sample_rate, channels=self.settings.channels, dtype="float32")
    sd.wait()
    return recording

def play(self, audio: np.ndarray) -> None:
    """Play float32 waveform."""
    sd.play(audio, samplerate=self.settings.sample_rate)
    sd.wait()

def save_wav(self, audio: np.ndarray, path: str | "np.ndarray") -> str:
    """Save audio to disk in WAV format."""
    sf.write(path, audio, self.settings.sample_rate)
    return str(path)

def load_wav(self, path: str) -> np.ndarray:
    """Load waveform as float32 numpy array."""
    data, sr = sf.read(path, dtype="float32")
    if sr != self.settings.sample_rate:
        if data.ndim > 1:
            data = np.mean(data, axis=1)
        data = librosa.resample(data, orig_sr=sr, target_sr=self.settings.sample_rate)
    return np.asarray(data, dtype=np.float32)


