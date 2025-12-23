import sounddevice as sd
import queue
import threading
from faster_whisper import WhisperModel
import language_tool_python

from config import CONFIG
from advanced_voice_mimic import VoiceCrystal
from aba_engine import ABAEngine
from routine_engine import RoutineEngine
from behavior_monitor import BehaviorMonitor

class SpeechLoop:
    def __init__(self):
        self.q = queue.Queue()
        self.vc = VoiceCrystal(CONFIG)
        self.whisper = WhisperModel("tiny.en", device="cpu", compute_type="int8")
        self.tool = language_tool_python.LanguageTool('en-US')
        self.aba = ABAEngine(self.vc, CONFIG, None)
        self.behavior = BehaviorMonitor()
        self.running = True

    def callback(self, indata, frames, time, status):
        if status:
            print(status)
        self.q.put(indata.copy())

    def run(self):
        print("Jackson’s Companion v10.0 — Listening forever in your own voice ❤️")
        with sd.InputStream(samplerate=16000, channels=1, dtype='float32', blocksize=512, callback=self.callback):
            buffer = np.array([], dtype='float32')
            while self.running:
                while not self.q.empty():
                    buffer = self.q.get()
                    buffer = np.append(buffer, data.flatten())

                # Autism-tuned VAD (very patient)
                if len(buffer) > 16000 * 10:  # max 10s
                    buffer = buffer[-16000*8:]

                rms = np.sqrt(np.mean(buffer[-16000*2:]**2)) if len(buffer) > 0 else 0
                if rms < 0.01 and len(buffer) > 16000 * 1.2:  # 1.2s silence = end of utterance
                    if len(buffer) > 16000 * 0.3:  # min speech length
                        self.process_utterance(buffer.copy())
                    buffer = np.array([], dtype='float32')

    def process_utterance(self, audio: np.ndarray):
        # 1. Transcribe
        segments, _ = self.whisper.transcribe(audio, language="en", vad_filter=True)
        raw_text = " ".join(s.text for s in segments).strip()
        if not raw_text:
            return

        # 2. Correct (only clarity/grammar, never meaning)
        corrected = self.tool.correct(raw_text)

        # 3. Behavior analysis
        style = "neutral"
        if "anxious" in self.behavior.current_state or "meltdown" in self.behavior.current_state:
            style = "calm"
        elif self.aba.success_streak >= 3:
            style = "excited"

        # 4. Echo back in child's exact voice with prosody transfer
        self.vc.say_in_child_voice(corrected, style=style, prosody_source=audio)

        # 5. ABA tracking
        self.aba.track_success_if_match(audio, corrected)

    def stop(self):
        self.running = False

