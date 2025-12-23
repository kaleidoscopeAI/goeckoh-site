import threading
import queue
import time
import numpy as np

# BULLETPROOF IMPORT: Graceful degradation
AUDIO_AVAILABLE = False
try:
    import sounddevice as sd
    AUDIO_AVAILABLE = True
except (ImportError, OSError):
    print("Warning: No Audio Device found. Running in Silent Mode.")

# Import Rust Kernel (Simulate Try/Except for uncompiled dev environments)
RUST_AVAILABLE = False
try:
    import bio_audio
    if hasattr(bio_audio, "BioAcousticEngine") or hasattr(bio_audio, "BioEngine"):
        RUST_AVAILABLE = True
except ImportError:
    print("Warning: Rust Kernel 'bio_audio' not compiled. Sound Synthesis Disabled.")

class AudioBridge:
    def __init__(self):
        self.q = queue.Queue(maxsize=5)
        self.running = True
        
        # Spin up Rust Engine if possible
        self.engine = None
        if RUST_AVAILABLE:
            try:
                if hasattr(bio_audio, "BioAcousticEngine"):
                    self.engine = bio_audio.BioAcousticEngine()
                else:
                    self.engine = bio_audio.BioEngine()
            except Exception as exc:
                print(f"Warning: Rust engine unavailable ({exc})")
        
        # Start Consumer Thread
        t = threading.Thread(target=self._worker, daemon=True)
        t.start()

    def trigger(self, text: str, arousal: float):
        """Non-blocking call to queue sound"""
        if RUST_AVAILABLE and self.engine:
            try:
                # Calls C-Level Rust Function
                pcm_data = self.engine.synthesize(len(text), arousal)
                # Drop frame if queue full (Better to skip audio than freeze UI)
                if not self.q.full():
                    self.q.put(pcm_data)
            except (ImportError, OSError, RuntimeError) as e:
                print(f"Synthesis Error: {e}")

    def _worker(self):
        """Threaded playback loop"""
        if not AUDIO_AVAILABLE:
            return

        while self.running:
            try:
                pcm_data = self.q.get(timeout=1)
                # Convert standard Vec<f32> to Numpy
                arr = np.array(pcm_data, dtype=np.float32)
                # Hardware write
                sd.play(arr, samplerate=22050, blocking=True)
            except queue.Empty:
                continue
            except (OSError, RuntimeError) as e:
                print(f"Playback Hardware Failure: {e}")
                # Don't crash loop, just sleep and retry
                time.sleep(1)
