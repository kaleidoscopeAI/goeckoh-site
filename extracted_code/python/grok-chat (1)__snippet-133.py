def __init__(self):
    self.q = queue.Queue()
    self.stop = threading.Event()

def callback(self, indata, *args):
    self.q.put(indata.copy().flatten())

def start(self):
    threading.Thread(target=self._run, daemon=True).start()

def _run(self):
    with sd.InputStream(samplerate=CONFIG.sample_rate, channels=1, dtype='float32', callback=self.callback):
        while not self.stop.is_set():
            sd.sleep(100)

def get(self) -> Optional[np.ndarray]:
    try:
        return self.q.get(timeout=0.1)
    except queue.Empty:
        return None

