def __init__(self, model_size: str = "small.en", vad_threshold: float = 0.45, min_silence_ms: int = 1200):
    self.model = WhisperModel(model_size, device="cpu", compute_type="int8")
    self.q = queue.Queue()
    self.stop = False
    self.vad_threshold = vad_threshold
    self.min_silence_ms = min_silence_ms
    self.sample_rate = 16000

def callback(self, indata: np.ndarray, *args):
    self.q.put(indata.flatten())

def start(self):
    threading.Thread(target=self._listen, daemon=True).start()

def _listen(self):
    with sd.InputStream(samplerate=self.sample_rate, channels=1, dtype='float32', callback=self.callback):
        while not self.stop:
            sd.sleep(100)

def transcribe_chunk(self) -> Optional[Tuple[str, np.ndarray]]:
    try:
        audio = self.q.get(timeout=0.1)
    except queue.Empty:
        return None

    # Simple VAD
    rms = np.sqrt(np.mean(audio**2))
    if rms < self.vad_threshold:
        return None

    segments, _ = self.model.transcribe(audio, vad_filter=True, vad_parameters={"threshold": self.vad_threshold, "min_silence_duration_ms": self.min_silence_ms})
    text = " ".join(s.text for s in segments).strip()
    return text, audio if text else None

