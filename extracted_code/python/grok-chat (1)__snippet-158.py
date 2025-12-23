def __init__(self, vad_threshold: float = 0.45, min_silence_ms: int = 1200):
    self.q = queue.Queue()
    self.stop = False
    self.vad_threshold = vad_threshold
    self.min_silence_ms = min_silence_ms
    self.sample_rate = 16000
    # No deps: Simple word list for "transcription" sim (improve with custom dict)
    self.word_dict = ["hello", "help", "happy", "voice", "echo"]  # Expand manually

def callback(self, indata: np.ndarray, *args):
    self.q.put(indata.flatten())

def start(self):
    import threading  # Built-in
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

    # Zero-dep VAD: RMS threshold
    rms = np.sqrt(np.mean(audio**2))
    if rms < self.vad_threshold:
        return None

    # Zero-dep "transcription": Simulate by matching audio length/peaks to words (placeholder; real needs model)
    # For improvement: Save to wave, analyze peaks
    text = random.choice(self.word_dict) if rms > 0.6 else ""  # Sim; replace with better heuristic

    return text, audio if text else None
