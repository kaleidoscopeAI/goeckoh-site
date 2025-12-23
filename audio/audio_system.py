import threading
import queue
import time
import numpy as np

# --- IMPORTS ---
AUDIO_AVAILABLE = False
RUST_AVAILABLE = False

try:
    import sounddevice as sd
    AUDIO_AVAILABLE = True
except ImportError:
    print("! SoundDevice library missing. Audio IO Disabled.")

try:
    import bio_audio
    # Prefer the modern BioAcousticEngine exported by rust_core
    if hasattr(bio_audio, "BioAcousticEngine"):
        RUST_AVAILABLE = True
except ImportError:
    print("! BioAudio Kernel missing. Physics simulation unavailable.")

class AudioSystem:
    def __init__(self):
        self.q = queue.Queue(maxsize=10) # Backpressure protection
        self.running = True
        self.bio_engine = None
        self.rust_available = RUST_AVAILABLE
        
        if self.rust_available:
            try:
                self.bio_engine = bio_audio.BioAcousticEngine()
            except Exception as exc:
                print(f"! BioAudio engine unavailable: {exc}")
                self.bio_engine = None
                self.rust_available = False
            
        self.thread = threading.Thread(target=self._playback_loop, daemon=True)
        self.thread.start()

    def enqueue_response(self, text: str, arousal: float):
        """Public API to add audio task."""
        if not self.rust_available or not self.bio_engine:
            return

        # Pure Synthesis Path
        # Generates float32 vector from Rust
        pcm_data = self.bio_engine.synthesize(len(text), float(arousal))
        
        # Convert to Numpy for sounddevice
        pcm_np = np.array(pcm_data, dtype=np.float32)
        
        try:
            self.q.put(pcm_np, timeout=0.5) # Non-blocking fail
        except queue.Full:
            pass # Drop frame if system overloaded (Real-time requirement)

    def _playback_loop(self):
        """Consumer Loop."""
        while self.running:
            if not AUDIO_AVAILABLE:
                time.sleep(1.0)
                continue

            try:
                # Wait for data
                data = self.q.get(timeout=1.0)
                sd.play(data, samplerate=22050, blocking=True)
                self.q.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                print(f"[AUDIO ERROR] Playback failed: {e}")
