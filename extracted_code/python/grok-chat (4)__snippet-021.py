import threading
import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel
import language_tool_python

from config import CONFIG
from advanced_voice_mimic import VoiceCrystal

class SpeechLoop:
    def __init__(self):
        self.q = queue.Queue()
        self.vc = VoiceCrystal(CONFIG)
        self.model = WhisperModel("tiny.en", device="cpu", compute_type="int8")
        self.tool = language_tool_python.LanguageTool('en-US')
        self.buffer = np.array([], dtype='float32')
        self.running = True

    def callback(self, indata, frames, time, status):
        self.q.put(indata.copy()[:, 0])

    def run(self):
        print("Companion active — listening forever in your own voice ❤️")
        with sd.InputStream(samplerate=16000, channels=1, dtype='float32', blocksize=1024, callback=self.callback):
            while self.buffer = np.array([], dtype='float32')
            while self.running:
                try:
                    chunk = self.q.get(timeout=1)
                    self.buffer = np.append(self.buffer, chunk)

                    # Detect end of utterance (1.5s silence)
                    if len(self.buffer) > 24000:
                        recent = self.buffer[-24000:]
                        if np.max(np.abs(recent)) < 0.01:
                            if len(self.buffer) > 8000:  # min length
                                audio = self.buffer.copy()
                                self.buffer = np.array([], dtype='float32')
                                threading.Thread(target=self.process, args=(audio,), daemon=True).start()
                except queue.Empty:
                    continue

    def process(self, audio: np.ndarray):
        segments, _ = self.model.transcribe(audio, language="en", vad_filter=True)
        raw = " ".join(seg.text for seg in segments).strip()
        if not raw:
            return

        corrected = self.tool.correct(raw)

        # Echo in child's exact voice with full prosody transfer
        self.vc.say(corrected, prosody_source=audio)

        # Harvest good attempts for lifelong drift (every clean repeat)
        # Simple clarity score — if very clear, save as new facet
        if np.mean(librosa.feature.rms(y=audio)) > 0.02:
            self.vc.harvest_facet(audio)

