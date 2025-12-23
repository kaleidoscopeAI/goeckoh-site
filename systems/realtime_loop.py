# goeckoh/realtime_loop.py
import json
import logging
import os
import queue
import time
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

import numpy as np

try:
    import sounddevice as sd
    AUDIO_AVAILABLE = True
except Exception as e:
    logging.warning("sounddevice unavailable: %s", e)
    AUDIO_AVAILABLE = False

try:
    from vosk import Model, KaldiRecognizer
    VOSK_AVAILABLE = True
except Exception as e:
    logging.warning("vosk unavailable: %s", e)
    VOSK_AVAILABLE = False

try:
    from goeckoh.systems.complete_unified_system import CompleteUnifiedSystem
except Exception as e:  # pragma: no cover - defensive import
    logging.exception("Failed to import CompleteUnifiedSystem; GUI will run in stub mode. %s", e)
    CompleteUnifiedSystem = None  # type: ignore


class EchoGoeckohSystem:
    """
    GUI/backend bridge that wraps the CompleteUnifiedSystem with real mic→STT→TTS.
    Provides the minimal API expected by the PySide6 controllers.
    """

    def __init__(self):
        self.mode = "hybrid"
        self._active_voice: Optional[str] = "default"
        self.core: Optional[CompleteUnifiedSystem] = None
        self.asr_model: Optional[Model] = None
        self.samplerate: int = 16000
        self.asr_available = False
        self._auto_running = False

        if CompleteUnifiedSystem is not None:
            try:
                self.core = CompleteUnifiedSystem()
                logging.info("Initialized CompleteUnifiedSystem backend for GUI.")
            except Exception as e:  # pragma: no cover - defensive
                logging.exception("Failed to initialize CompleteUnifiedSystem; falling back to stub. %s", e)
        else:
            logging.warning("CompleteUnifiedSystem unavailable; using stub backend.")

        # Load offline STT (Vosk) if available
        model_path = self._auto_find_vosk_model()
        if VOSK_AVAILABLE and model_path:
            try:
                self.asr_model = Model(str(model_path))
                if AUDIO_AVAILABLE:
                    try:
                        self.samplerate = int(sd.query_devices(None, "input")["default_samplerate"])
                    except Exception:
                        self.samplerate = 16000
                self.asr_available = True
                logging.info("Vosk model loaded for real-time STT: %s", model_path)
            except Exception as e:
                logging.exception("Failed to load Vosk model at %s: %s", model_path, e)
                self.asr_available = False
        else:
            if not VOSK_AVAILABLE:
                logging.warning("Vosk not installed; live STT disabled.")
            else:
                logging.warning("Vosk model not found; live STT disabled (will pass through audio).")

    def _auto_find_vosk_model(self) -> Optional[Path]:
        """Look for a bundled Vosk model under ./models or a common path."""
        candidates = []
        base = Path(__file__).resolve().parents[2] / "models"
        if base.exists():
            for p in base.iterdir():
                if p.is_dir() and p.name.startswith("vosk-model"):
                    candidates.append(p)
        env_path = os.getenv("VOSK_MODEL_PATH")
        if env_path:
            p = Path(env_path).expanduser()
            if p.exists():
                candidates.insert(0, p)
        return candidates[0] if candidates else None

    def _transcribe_once(self, seconds: float) -> Tuple[str, Optional[np.ndarray]]:
        """
        Capture audio for a short window and return (text, audio_pcm_float32).
        Uses streaming Vosk to reduce latency and keep dependencies minimal.
        """
        if not AUDIO_AVAILABLE:
            return "", None

        duration = max(0.5, seconds)
        q: "queue.Queue[bytes]" = queue.Queue()
        buffer = bytearray()

        def _cb(indata, frames, callback_time, status):  # type: ignore[override]
            if status:
                logging.debug("Input status: %s", status)
            q.put(bytes(indata))

        try:
            rec = KaldiRecognizer(self.asr_model, self.samplerate) if (self.asr_available and self.asr_model) else None
            partial_text = ""

            with sd.RawInputStream(
                samplerate=self.samplerate,
                blocksize=8000,
                dtype="int16",
                channels=1,
                callback=_cb,
            ):
                end_ts = time.time() + duration
                while time.time() < end_ts:
                    try:
                        chunk = q.get(timeout=0.1)
                    except queue.Empty:
                        continue
                    buffer.extend(chunk)
                    if rec:
                        if rec.AcceptWaveform(chunk):
                            result = json.loads(rec.Result())
                            partial_text = result.get("text", partial_text)
                        else:
                            partial = json.loads(rec.PartialResult()).get("partial")
                            if partial:
                                partial_text = partial
                if rec:
                    result = json.loads(rec.FinalResult())
                    final_text = result.get("text", "").strip()
                    text = final_text or partial_text
                else:
                    text = ""
        except Exception as e:
            logging.exception("Streaming Vosk recognition failed: %s", e)
            return "", None

        audio_f32: Optional[np.ndarray] = None
        try:
            audio_i16 = np.frombuffer(buffer, dtype=np.int16)
            audio_f32 = audio_i16.astype(np.float32) / 32768.0
        except Exception:
            audio_f32 = None

        return text.strip(), audio_f32

    def _process(self, text: str, audio_input: Optional[np.ndarray] = None) -> Tuple[str, str, dict]:
        """
        Run the full pipeline and return (raw, corrected) text pair for the GUI.
        The "corrected" text is the synthesized response from the unified system.
        """
        if not self.core:
            # Stub mode
            return text, f"[stub-response] {text}", {"engine": "stub", "clone": False}

        result = self.core.process_input(text_input=text, audio_input=audio_input)
        response = result.get("response_text", "")
        sys_status = result.get("system_status", {}) if isinstance(result, dict) else {}
        status = {
            "engine": result.get("system_status", {}).get("audio_engine", "unknown"),
            "clone": bool(getattr(self.core, "clone_ref_wav", None)),
            "behavior_event": result.get("behavior_event"),
            "processing_time": result.get("processing_time"),
            "gcl": sys_status.get("gcl"),
            "stress": sys_status.get("stress"),
            "system_mode": sys_status.get("system_mode"),
            "coaching": result.get("coaching") or sys_status.get("coaching"),
            "emotional_state": sys_status.get("emotional_state"),
        }
        return text, response, status

    def loop_once(self, seconds: float = 0.8, voice_name: Optional[str] = None):
        """
        Capture mic for a short window, correct, and play back in near-real time.
        """
        raw_text, audio_input = self._transcribe_once(seconds)
        if not raw_text:
            raw_text = "[silence]"
        return self._process(raw_text, audio_input=audio_input)

    def speak_text(self, text: str, voice_name: Optional[str] = None):
        return self._process(text, audio_input=None)

    def start_auto_loop(self, seconds: float = 0.8, on_result=None, on_status=None):
        """
        Continuously capture → correct → play back in a background loop.
        on_result: optional callback(raw, corrected) for UI updates.
        on_status: optional callback(status_dict) for engine/clone/latency updates.
        """
        if self._auto_running:
            return
        self._auto_running = True

        def _run():
            while self._auto_running:
                try:
                    raw_text, audio_input = self._transcribe_once(seconds)
                    if not raw_text:
                        raw_text = "[silence]"
                    raw, corrected, status = self._process(raw_text, audio_input=audio_input)
                    if on_result:
                        on_result((raw, corrected))
                    if on_status:
                        on_status(status)
                except Exception as e:
                    logging.exception("Auto loop error: %s", e)
                # minimal pause to avoid tight loop when mic idle
                time.sleep(0.05)

        import threading

        t = threading.Thread(target=_run, daemon=True)
        t.start()

    def stop_auto_loop(self):
        self._auto_running = False

    @staticmethod
    def enroll_voice_xtts(name: str, wav_path: str, language: str = "en"):
        logging.info("XTTS enrollment requested (GUI): name=%s, wav=%s, lang=%s", name, wav_path, language)

    @staticmethod
    def enroll_voice_math(name: str, wav_path: str):
        logging.info("Math voice enrollment requested (GUI): name=%s, wav=%s", name, wav_path)

    def set_clone_wav(self, path: str):
        """Update clone reference WAV for downstream synthesis."""
        if self.core and hasattr(self.core, "set_clone_wav"):
            self.core.set_clone_wav(path)
        else:
            logging.warning("Clone WAV set while core unavailable; request ignored.")

    @property
    def voice_name(self):
        return self._active_voice

    @voice_name.setter
    def voice_name(self, value):
        self._active_voice = value
