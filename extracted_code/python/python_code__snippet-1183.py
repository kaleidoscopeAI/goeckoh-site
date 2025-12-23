import sounddevice as sd
from vosk import Model, KaldiRecognizer
import queue
import json
import threading
import os
import sys

class SpeechProcessor:
    """
    Handles real-time, offline speech-to-text processing in a background thread.
    """
    def __init__(self, model_path: str, text_callback, status_callback):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Vosk model not found at path: {model_path}")

        self.model = Model(model_path)
        self.samplerate = int(sd.query_devices(None, 'input')['default_samplerate'])
        
        self.q = queue.Queue()
        self.text_callback = text_callback  # Called with final transcribed text
        self.status_callback = status_callback # Called with status updates
        
        self.running = True
        self.thread = threading.Thread(target=self._recognizer_loop, daemon=True)

    def _audio_callback(self, indata, frames, time, status):
        """This is called (from a separate thread) for each audio block."""
        if status:
            print(status, file=sys.stderr)
        self.q.put(bytes(indata))

    def _recognizer_loop(self):
        """
        Continuously processes audio from the queue and uses Vosk to recognize speech.
        """
        recognizer = KaldiRecognizer(self.model, self.samplerate)
        recognizer.SetWords(True)
        self.status_callback("LISTENING")

        while self.running:
            try:
                data = self.q.get(timeout=1.0)
                if recognizer.AcceptWaveform(data):
                    result_json = recognizer.Result()
                    result = json.loads(result_json)
                    if result.get("text"):
                        self.status_callback("PROCESSING")
                        self.text_callback(result["text"])
                        self.status_callback("LISTENING")
            except queue.Empty:
                continue
            except Exception as e:
                self.status_callback("ERROR")
                print(f"[SpeechProcessor Error] {e}")

    def start(self):
        """Starts the audio stream and recognition thread."""
        try:
            self.stream = sd.RawInputStream(
                samplerate=self.samplerate, 
                blocksize=8000, 
                device=None, 
                dtype='int16',
                channels=1, 
                callback=self._audio_callback
            )
            self.stream.start()
            self.thread.start()
            print("[SpeechProcessor] Started successfully.")
        except Exception as e:
            self.status_callback("ERROR")
            print(f"[SpeechProcessor] Failed to start audio stream: {e}")

    def stop(self):
        """Stops the audio stream and recognition thread."""
        self.running = False
        if hasattr(self, 'stream'):
            self.stream.stop()
            self.stream.close()
        self.thread.join(timeout=2.0)
        print("[SpeechProcessor] Stopped.")


