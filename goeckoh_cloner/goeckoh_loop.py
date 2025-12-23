# goeckoh_loop.py
import io
import queue
import threading
import time
from pathlib import Path
from typing import Optional

import numpy as np
import sounddevice as sd
import soundfile as sf
import webrtcvad
import whisper

from config import SAMPLE_RATE, DEFAULT_VOICE_LABEL, WHISPER_MODEL_NAME
from voice_profile import VoiceFingerprint, get_profile_path
from correction_engine import clean_asr_text
from bubble_synthesizer import feed_text_through_bubble
from visual_engine_bridge import BubbleBroadcaster, compute_bubble_state
from cloning_backend import synthesize_text_with_clone
from licensing_security import check_license_valid


class AudioCapture:
    def __init__(self, sample_rate: int = SAMPLE_RATE, frame_ms: int = 30):
        self.sample_rate = sample_rate
        self.frame_ms = frame_ms
        self.q = queue.Queue()
        self.stream: Optional[sd.InputStream] = None

    def _callback(self, indata, frames, time_info, status):
        if status:
            print("[AudioCapture] Status:", status)
        self.q.put(indata[:, 0].copy())

    def start(self, device: Optional[int] = None):
        self.stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype="float32",
            callback=self._callback,
            device=device,
        )
        self.stream.start()
        print("[AudioCapture] Started.")

    def stop(self):
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None
        print("[AudioCapture] Stopped.")

    def read_frame(self, timeout: float = 1.0) -> Optional[np.ndarray]:
        try:
            return self.q.get(timeout=timeout)
        except queue.Empty:
            return None


class VADUtteranceDetector:
    def __init__(self, sample_rate: int = SAMPLE_RATE, frame_ms: int = 30, aggressiveness: int = 2):
        self.vad = webrtcvad.Vad(aggressiveness)
        self.sample_rate = sample_rate
        self.frame_ms = frame_ms
        self.frame_samples = int(sample_rate * frame_ms / 1000.0)
        self.min_active_frames = int(300 / frame_ms)    # ~300ms
        self.silence_end_frames = int(800 / frame_ms)   # ~800ms

    def detect_utterance(self, capture: AudioCapture, stop_event: threading.Event) -> Optional[np.ndarray]:
        frames = []
        active_count = 0
        silence_count = 0
        print("[VAD] Waiting for speech...")

        while not stop_event.is_set():
            frame = capture.read_frame(timeout=1.0)
            if frame is None:
                continue

            pcm16 = (frame * 32767.0).astype(np.int16).tobytes()
            is_speech = self.vad.is_speech(pcm16, self.sample_rate)

            if is_speech:
                active_count += 1
                silence_count = 0
                frames.append(frame)
            else:
                if active_count == 0:
                    continue
                silence_count += 1
                frames.append(frame)

            if active_count >= self.min_active_frames and silence_count >= self.silence_end_frames:
                break

        if not frames:
            return None
        utterance = np.concatenate(frames, axis=0)
        print(f"[VAD] Captured utterance, {len(utterance) / self.sample_rate:.2f}s")
        return utterance


class GoeckohLoop:
    def __init__(self, profile_name: str, stop_event: Optional[threading.Event] = None):
        if not check_license_valid():
            raise RuntimeError("No valid license found. Please activate first.")

        profile_path = get_profile_path(profile_name)
        if not profile_path.exists():
            raise FileNotFoundError(f"No VoiceFingerprint found at {profile_path}. Run voice_logger.py first.")
        self.profile = VoiceFingerprint.from_json(profile_path)
        self.voice_label = profile_name or DEFAULT_VOICE_LABEL

        self.capture = AudioCapture()
        self.vad = VADUtteranceDetector()
        self.ws = BubbleBroadcaster()

        print(f"[Goeckoh] Loading Whisper model '{WHISPER_MODEL_NAME}'...")
        self.whisper_model = whisper.load_model(WHISPER_MODEL_NAME)

        self.stop_event = stop_event or threading.Event()

    def _play_audio(self, audio_bytes: bytes):
        y, sr = sf.read(io.BytesIO(audio_bytes), always_2d=False)
        sd.play(y, samplerate=sr)
        sd.wait()

    def _run_bubble_animation(self, control_curves: dict):
        t = control_curves["t"]
        n = len(t)
        for i in range(n):
            if self.stop_event.is_set():
                break
            state = compute_bubble_state(self.profile, control_curves, i, idle=False)
            self.ws.send_state(state)
            if i < n - 1:
                dt = t[i + 1] - t[i]
                time.sleep(max(dt, 0.005))

    def _idle_heartbeat(self):
        base_radius = self.profile.base_radius
        hue = float((self.profile.mu_f0 - 80.0) / (400.0 - 80.0))
        while not self.stop_event.is_set():
            for phase in np.linspace(0, 2 * np.pi, 60):
                if self.stop_event.is_set():
                    break
                radius = base_radius * (1.0 + 0.05 * np.sin(phase))
                state = {
                    "radius": float(radius),
                    "spike": float(self.profile.base_sharpness * 0.5),
                    "metalness": float(self.profile.base_metalness),
                    "roughness": float(self.profile.base_roughness),
                    "hue": hue,
                    "halo": 0.15,   # gentle ambient presence
                    "idle": True,
                }
                self.ws.send_state(state)
                time.sleep(0.1)

    def start(self):
        # idle heartbeat
        threading.Thread(target=self._idle_heartbeat, daemon=True).start()
        self.capture.start()
        print("[Goeckoh] Loop started. Speak when ready.")

        try:
            while not self.stop_event.is_set():
                utterance = self.vad.detect_utterance(self.capture, self.stop_event)
                if utterance is None or self.stop_event.is_set():
                    continue

                # 1) save temp, transcribe
                tmp_path = Path("tmp_utterance.wav")
                sf.write(str(tmp_path), utterance, SAMPLE_RATE)
                print("[Whisper] Transcribing utterance...")
                result = self.whisper_model.transcribe(str(tmp_path), language="en")
                raw_text = (result.get("text") or "").strip()
                print(f"[Whisper] Raw: {raw_text!r}")

                # 2) first-person correction
                corrected = clean_asr_text(raw_text)
                print(f"[Corrected] -> {corrected!r}")
                if not corrected:
                    continue

                # 3) psychoacoustic control curves
                control_curves = feed_text_through_bubble(corrected, self.profile)

                # 4) voice-cloned synthesis
                audio_bytes = synthesize_text_with_clone(
                    corrected,
                    voice_label=self.voice_label,
                    accent="en-newest",
                    speed=1.0,
                    watermark=" @Bubble",
                )

                # 5) animate + playback
                anim_thread = threading.Thread(
                    target=self._run_bubble_animation,
                    args=(control_curves,),
                    daemon=True,
                )
                anim_thread.start()
                self._play_audio(audio_bytes)
                anim_thread.join()

        finally:
            self.capture.stop()
            print("[Goeckoh] Loop stopped.")


if __name__ == "__main__":
    import argparse

    from licensing_security import check_license_valid

    if not check_license_valid():
        raise SystemExit("No valid license. Run main_gui.py to activate.")

    parser = argparse.ArgumentParser(description="Run the Goeckoh Neuro-Acoustic Cloner loop.")
    parser.add_argument("--name", required=True, help="Profile name used during enrollment")
    args = parser.parse_args()

    loop = GoeckohLoop(args.name)
    loop.start()
