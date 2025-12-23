from __future__ import annotations

import numpy as np
import torch
from faster_whisper import WhisperModel

from core.settings import SpeechSettings


class STT:
    """
    Speech-to-Text using faster-whisper.
    """

    def __init__(self, settings: SpeechSettings):
        self.settings = settings
        self.model = WhisperModel(
            self.settings.whisper_model, device="cpu", compute_type="int8"
        )

    def transcribe(self, audio_chunk: np.ndarray) -> str:
        """
        Transcribes an audio chunk and returns the text.
        """
        segments, _ = self.model.transcribe(audio_chunk, vad_filter=False)
        return "".join(s.text for s in segments).strip()


