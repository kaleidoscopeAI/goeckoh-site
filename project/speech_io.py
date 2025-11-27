# echo_core/speech_io.py
from __future__ import annotations
import queue
import threading
import numpy as np
import sounddevice as sd
import torch
import time
from typing import Optional
from config import CONFIG

# Silero VAD model - loaded on first use
_SILERO_MODEL = None
_SILERO_SAMPLE_RATE = 16000

def _load_vad_model() -> None:
    """Loads the Silero VAD model. Handles potential download errors."""
    global _SILERO_MODEL
    if _SILERO_MODEL is None:
        try:
            model, _ = torch.hub.load(
                repo_or_dir="snakers4/silero-vad",
                model="silero_vad",
                force_reload=False,
                trust_repo=True  # Trust the official repo
            )
            _SILERO_MODEL = model
        except Exception as e:
            print(f"Error loading Silero VAD model: {e}. Voice activity detection may not work.")
            _SILERO_MODEL = None

def get_speech_prob(audio: np.ndarray) -> float:
    """
    Gets the speech probability of an audio chunk using the Silero VAD model.
    Fails safely by returning a high probability if the model is not available.
    """
    _load_vad_model()
    if _SILERO_MODEL is None:
        # If model failed to load, assume it is speech to avoid dropping audio
        return 1.0
    try:
        with torch.no_grad():
            tensor = torch.from_numpy(audio.astype("float32"))
            prob = _SILERO_MODEL(tensor, _SILERO_SAMPLE_RATE).item()
        return prob
    except Exception as e:
        print(f"Error during VAD inference: {e}")
        return 1.0 # Fail safely by assuming it is speech

class AudioStream:
    """
    Manages the microphone input stream in a separate thread to avoid
    blocking the main application loop.
    """
    def __init__(self) -> None:
        self._q: "queue.Queue[np.ndarray]" = queue.Queue()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def _callback(self, indata, frames, time_info, status):
        """This is called by sounddevice for each new audio chunk."""
        if status:
            print(f"AudioStream status: {status}")
        self._q.put(indata.copy())

    def start(self) -> None:
        """Starts the audio stream thread."""
        if self._thread is not None and self._thread.is_alive():
            return

        def run():
            try:
                with sd.InputStream(
                    channels=1,
                    samplerate=CONFIG.audio.sample_rate,
                    blocksize=CONFIG.audio.block_size,
                    callback=self._callback,
                    dtype="float32"
                ):
                    while not self._stop.is_set():
                        time.sleep(0.01)
            except Exception as e:
                print(f"Fatal error in AudioStream thread: {e}")

        self._stop.clear()
        self._thread = threading.Thread(target=run, daemon=True)
        self._thread.start()
        print("AudioStream started.")

    def stop(self) -> None:
        """Stops the audio stream thread."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        print("AudioStream stopped.")

    def get_chunk(self, timeout: float = 1.0) -> np.ndarray | None:
        """Retrieves an audio chunk from the internal queue."""
        try:
            return self._q.get(timeout=timeout)
        except queue.Empty:
            return None
