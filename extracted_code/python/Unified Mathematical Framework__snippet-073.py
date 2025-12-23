import queue
import threading
import time
import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel
import language_tool_python

from config import CONFIG
from advanced_voice_mimic import VoiceCrystal
from aba_engine import ABAEngine

class SpeechLoop:
    def __init__(self):
        self.audio_queue = queue.Queue(maxsize=100)
        self.vc = VoiceCrystal(CONFIG)
        self.whisper = WhisperModel("tiny.en", device="cpu", compute_type="int8")
        self.tool = language_tool_python.LanguageTool('en-US')
        self.aba = ABAEngine(self.vc, CONFIG)

        self.buffer = np.array([], dtype='float32')
        self.is_listening = True

    def audio_callback(self, indata, frames, time_info, status):
        if status:
            print("Audio status:", status)
        self.audio_queue.put(indata.copy()[:, 0])  # mono

    def process_utterance(self, audio: np.ndarray):
        if len(audio) < 16000 * 0.4:  # min 0.4s speech
            return

        # 1. Transcribe raw utterance
        segments, _ = self.whisper.transcribe(audio, language="en", vad_filter=True)
        raw_text = " ".join(seg.text for seg in segments).strip().lower()
        if not raw_text:
            return

        # 2. Correct text (clarity + grammar only)
        corrected_text = self.tool.correct(raw_text)

        # 3. Echo in child's exact voice + prosody transfer from original
        self.vc.say_in_child_voice(corrected_text, style="neutral", prosody_source=audio)

        # 4. Wait for child's repeat (5 second window)
        repeat_audio = self.capture_silence_limited(max_seconds=5.0)
        if repeat_audio is not None and len(repeat_audio) > 16000 * 0.4:
            # Transcribe repeat
            segs, _ = self.whisper.transcribe(repeat_audio, language="en")
            repeat_text = " ".join(s.text for s in segs).strip().lower()

            # Simple but highly effective match (Levenshtein ratio)
            from Levenshtein import ratio  # pip install python-Levenshtein or use similar
            similarity = ratio(repeat_text, corrected_text)
            if similarity >= 0.95 or repeat_text == corrected_text:
                # SUCCESS → harvest repeat as new voice facet + ABA tracking
                style = "neutral"  # or detect energy from repeat_audio
                self.vc.add_facet(repeat_audio, style=style)
                self.aba.track_skill_progress("speech_clarity", success=True)
                self.aba.reinforce_success(corrected_text)
            else:
                self.aba.track_skill_progress("speech_clarity", success=False)

    def capture_silence_limited(self, max_seconds: float = 5.0) -> np.ndarray | None:
        start = time.time()
        temp_buffer = np.array([], dtype='float32')
        while time.time() - start < max_seconds:
            try:
                data = self.audio_queue.get(timeout=0.1)
                temp_buffer = np.append(temp_buffer, data)
                rms = np.sqrt(np.mean(data**2))
                if rms < 0.01:  # silence
                    if len(temp_buffer) > 16000 * 0.4:
                        return temp_buffer
            except queue.Empty:
                continue
        return temp_buffer if len(temp_buffer) > 16000 * 0.3 else None

    def run_forever(self):
        print("Jackson’s Companion — Eternal Listener Active ❤️")
        with sd.InputStream(samplerate=16000, channels=1, dtype='float32',
                           blocksize=1024, callback=self.audio_callback):
            while self.is_listening:
                try:
                    data = self.audio_queue.get(timeout=0.5)
                    self.buffer = np.append(self.buffer, data)

                    # Check for end of utterance (1.2s silence in last 2s)
                    if len(self.buffer) > 16000 * 2:
                        recent = self.buffer[-32000:]
                        if np.sqrt(np.mean(recent**2)) < 0.01:
                            utterance = self.buffer.copy()
                            self.buffer = np.array([], dtype='float32')  # reset
                            threading.Thread(target=self.process_utterance, args=(utterance,), daemon=True).start()
                except queue.Empty:
                    continue

