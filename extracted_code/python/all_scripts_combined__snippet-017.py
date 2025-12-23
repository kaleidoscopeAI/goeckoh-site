"""Handles all low-level audio capture and playback."""

def __init__(self, settings: AudioSettings):
    self.settings = settings
    self._q: queue.Queue[np.ndarray] = queue.Queue()
    self._stop_event = threading.Event()
    self._thread: threading.Thread | None = None

def _callback(self, indata: np.ndarray, frames: int, time, status) -> None:
    """This is called (from a separate thread) for each audio block."""
    if status:
        print(status)
    self._q.put(indata.copy())

def listen(self) -> None:
    """Starts the audio stream."""
    self._thread = threading.Thread(target=self._listen_thread)
    self._thread.start()

def _listen_thread(self) -> None:
    with sd.InputStream(
        samplerate=self.settings.sample_rate,
        channels=self.settings.channels,
        dtype="float32",
        callback=self._callback,
    ):
        while not self._stop_event.is_set():
            sd.sleep(1000)

def stop(self) -> None:
    """Stops the audio stream."""
    self._stop_event.set()
    if self._thread:
        self._thread.join()

def chunk_generator(self) -> Iterator[np.ndarray]:
    """Yields audio chunks of a specific size."""
    while not self._stop_event.is_set():
        try:
            yield self._q.get(timeout=1)
        except queue.Empty:
            continue

@staticmethod
def play(audio: np.ndarray, sample_rate: int) -> None:
    """Plays an audio chunk."""
    sd.play(audio, sample_rate)
    sd.wait()

