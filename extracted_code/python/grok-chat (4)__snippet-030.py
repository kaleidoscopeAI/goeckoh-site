import threading
import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel
import language_tool_python
from Levenshtein import ratio

from config import CONFIG
from advanced_voice_mimic import VoiceCrystal
from aba_engine import ABAEngine

class SpeechLoop:
    def __init__(self):
        self.q = queue.Queue(maxsize=200)
        self.vc = VoiceCrystal(CONFIG)
        self.model = WhisperModel("tiny.en", device="cpu", compute_type="int8")
        self.tool = language_tool_python.LanguageTool('en-US')
        self.aba = ABAEngine(self.vc, CONFIG)
        self.buffer = np.array([], dtype='float32')
        self.running = True

    def callback(self, indata, frames, time_info, status):
        if status:
            print("Audio error:", status)
        self.q.put(indata.copy()[:, 0])

    def run_forever(self):
        print("Jackson’s Companion v12.0 — Listening in your own voice forever ❤️")
        with sd.InputStream(samplerate=CONFIG.sample_rate, channels=1, dtype='float32',
                          blocksize=1024, callback=self.callback):
            while self.running:
                try:
                    chunk = self.q.get(timeout=0.5)
                    self.buffer = np.append(self.buffer, chunk)

                    if len(self.buffer) > CONFIG.sample_rate * 10:
                        self.buffer = self.buffer[-CONFIG.sample_rate * 8:]

                    if len(self.buffer) > CONFIG.sample_rate * 2:
                        recent = self.buffer[-int(CONFIG.sample_rate * CONFIG.silence_duration_seconds):]
                        if np.max(np.abs(recent)) < CONFIG.silence_threshold:
                            if len(self.buffer) >= CONFIG.sample_rate * CONFIG.min_utterance_seconds:
                                audio = self.buffer.copy()
                                self.buffer = np.array([], dtype='float32')
                                threading.Thread(target=self.process_utterance, args=(audio,), daemon=True).start()
                except queue.Empty:
                    continue
                except Exception as e:
                    print("Loop error:", e)

    def process_utterance(self, audio: np.ndarray):
        try:
            segments, _ = self.model.transcribe(audio, language="en", vad_filter=True, beam_size=5)
            raw_text = " ".join(seg.text for seg in segments).strip().lower()
            if not raw_text:
                return

            corrected_text = self.tool.correct(raw_text)

            style = "neutral"
            if self.aba.current_emotion in ["anxious", "high_energy"]:
                style = "calm"
            elif self.aba.success_streak >= 3:
                style = "excited"

            self.vc.say(corrected_text, style=style, prosody_source=audio)

            # Wait up to 6 seconds for repeat
            repeat_audio = self.wait_for_repeat()
            if repeat_audio is not None:
                segs, _ = self.model.transcribe(repeat_audio, language="en")
                repeat_text = " ".join(s.text for s in segs).strip().lower()
                if ratio(repeat_text, corrected_text) >= 0.94:
                    self.vc.harvest_facet(repeat_audio, style)
                    self.aba.track_skill_progress("articulation", success=True)
                    if self.aba.success_streak >= 3:
                        self.vc.say("I did it perfectly!", style="excited")
                else:
                    self.aba.track_skill_progress("articulation", success=False)
        except Exception as e:
            print("Processing error:", e)

    def wait_for_repeat(self, timeout=6.0) -> np.ndarray | None:
        start = time.time()
        repeat_buffer = np.array([], dtype='float32')
        while time.time() - start < timeout:
            try:
                chunk = self.q.get(timeout=0.1)
                repeat_buffer = np.append(repeat_buffer, chunk)
                if np.max(np.abs(chunk)) > 0.02:
                    continue
                if len(repeat_buffer) > CONFIG.sample_rate * 1.5 and np.max(np.abs(repeat_buffer[-int(CONFIG.sample_rate*1.0):])) < CONFIG.silence_threshold:
                    if len(repeat_buffer) >= CONFIG.sample_rate * CONFIG.min_utterance_seconds:
                        return repeat_buffer
            except queue.Empty:
                continue
        return repeat_buffer if len(repeat_buffer) > CONFIG.sample_rate * 0.5 else None

