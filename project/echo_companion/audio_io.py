import queue
import threading
import time
from typing import List, Optional

import numpy as np
import sounddevice as sd
import webrtcvad

from config import SAMPLE_RATE, CHANNELS, FRAME_DURATION_MS, VAD_SILENCE_MS, VAD_AGGRESSIVENESS


class AudioIO:
    """
    Handles microphone capture until silence using WebRTC VAD,
    and waveform playback.
    """

    def __init__(self) -> None:
        self.vad = webrtcvad.Vad(VAD_AGGRESSIVENESS)

    def _frame_generator(self, frame_duration_ms: int, audio: bytes):
        n = int(SAMPLE_RATE * (frame_duration_ms / 1000.0) * 2)  # 16-bit mono
        offset = 0
        while offset + n <= len(audio):
            yield audio[offset : offset + n]
            offset += n

    def _record_raw(self, max_seconds: float) -> bytes:
        """Record raw PCM16 bytes from microphone."""
        frames = []
        duration = max_seconds
        num_frames = int((duration * 1000) / FRAME_DURATION_MS)
        done = threading.Event()
        q_frames: queue.Queue[np.ndarray] = queue.Queue()

        def callback(indata, frames_count, time_info, status):
            if status:
                print("Audio callback status:", status)
            q_frames.put(indata.copy())
            if done.is_set():
                raise sd.CallbackStop()

        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype="int16",
            callback=callback,
        ):
            start = time.time()
            for _ in range(num_frames):
                try:
                    data = q_frames.get(timeout=duration)
                except queue.Empty:
                    break
                frames.append(data)
                if time.time() - start > duration:
                    done.set()
                    break

        return b"".join(f.tobytes() for f in frames)

    def record_utterance(self) -> Optional[np.ndarray]:
        """
        Record an utterance from the mic until we see enough consecutive
        non-speech frames from VAD, or we hit a hard time cap.
        Returns float32 waveform in range [-1, 1] or None if nothing captured.
        """
        raw_pcm = self._record_raw(max_seconds=12.0)
        if not raw_pcm:
            return None

        frames = list(self._frame_generator(FRAME_DURATION_MS, raw_pcm))

        speech_frames: List[bytes] = []
        silence_run_ms = 0
        saw_speech = False

        for frame in frames:
            is_speech = self.vad.is_speech(frame, SAMPLE_RATE)
            if is_speech:
                speech_frames.append(frame)
                silence_run_ms = 0
                saw_speech = True
            else:
                if saw_speech:
                    silence_run_ms += FRAME_DURATION_MS
                    if silence_run_ms >= VAD_SILENCE_MS:
                        break

        if not speech_frames:
            return None

        pcm_concat = b"".join(speech_frames)
        audio_int16 = np.frombuffer(pcm_concat, dtype=np.int16)
        audio_float = audio_int16.astype(np.float32) / 32768.0
        return audio_float

    def play_audio(self, audio: np.ndarray, sample_rate: int) -> None:
        """Play a float32 waveform in [-1, 1] at the given sample rate."""
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        sd.play(audio, samplerate=sample_rate)
        sd.wait()
