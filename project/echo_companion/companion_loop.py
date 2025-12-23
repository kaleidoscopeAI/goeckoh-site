import time

from audio_io import AudioIO
from asr_engine import ASREngine
from broken_speech_normalizer import normalize as normalize_broken
from voice_clone_engine import VoiceCloneEngine
from metrics_store import MetricsStore
from config import RTVC_SAMPLE_RATE, MAX_ALLOWED_LATENCY_SEC


class SpeechCompanion:
    """
    Real-time loop:
    mic -> VAD -> Whisper ASR -> broken speech normalize -> RTVC clone -> playback.
    """

    def __init__(self) -> None:
        self.audio_io = AudioIO()
        self.asr = ASREngine()
        self.clone_engine = VoiceCloneEngine()
        self.metrics = MetricsStore()

    def ensure_enrolled(self) -> None:
        if not self.clone_engine.is_enrolled():
            print("Enrolling speaker from reference directory...")
            self.clone_engine.enroll_from_directory()
            print("Enrollment complete.")

    def process_once(self) -> bool:
        """
        Capture one utterance, process, and speak back.
        Returns False if no utterance was captured.
        """
        print("\n[Companion] Listening for speech...")
        start_time = time.time()
        audio = self.audio_io.record_utterance()
        if audio is None:
            print("[Companion] No speech detected.")
            return False

        capture_done = time.time()

        raw_text = self.asr.transcribe(audio) or ""
        asr_done = time.time()
        if not raw_text.strip():
            print("[Companion] No text recognized.")
            return False

        print(f"[Companion] Heard: {raw_text!r}")

        normalization = normalize_broken(raw_text)
        normalized_text = normalization["normalized_text"]
        print(f"[Companion] Interpreted as: {normalized_text!r}")

        echo_text = normalized_text

        tts_start = time.time()
        cloned_audio = self.clone_engine.speak(echo_text)
        tts_done = time.time()

        self.audio_io.play_audio(cloned_audio, RTVC_SAMPLE_RATE)
        playback_done = time.time()

        total_latency = playback_done - start_time
        print(f"[Companion] Total latency: {total_latency:.2f}s")

        self.metrics.log(
            {
                "raw_text": raw_text,
                "normalized_text": normalized_text,
                "intent": normalization["intent"],
                "sentiment": normalization["sentiment"],
                "latency_sec": total_latency,
                "asr_time_sec": asr_done - capture_done,
                "tts_time_sec": tts_done - tts_start,
                "capture_time_sec": capture_done - start_time,
                "over_latency_budget": total_latency > MAX_ALLOWED_LATENCY_SEC,
            }
        )

        return True

    def loop_forever(self) -> None:
        self.ensure_enrolled()
        print("[Companion] Ready. Speak whenever you like. Ctrl+C to quit.")
        try:
            while True:
                self.process_once()
        except KeyboardInterrupt:
            print("\n[Companion] Shutting down.")
