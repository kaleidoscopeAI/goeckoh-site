from typing import Optional

import numpy as np
from faster_whisper import WhisperModel

from config import WHISPER_MODEL_SIZE, WHISPER_DEVICE, WHISPER_COMPUTE_TYPE


class ASREngine:
    def __init__(self) -> None:
        self.model = WhisperModel(
            WHISPER_MODEL_SIZE,
            device=WHISPER_DEVICE,
            compute_type=WHISPER_COMPUTE_TYPE,
        )

    def transcribe(self, audio: np.ndarray) -> Optional[str]:
        """
        Transcribe a single utterance waveform (float32 [-1,1]).
        Returns a text string or None if no text recognized.
        """
        if audio is None or len(audio) == 0:
            return None

        segments, _info = self.model.transcribe(
            audio,
            language="en",
            beam_size=5,
            vad_filter=False,  # VAD already applied
        )

        texts = [seg.text.strip() for seg in segments if seg.text.strip()]
        if not texts:
            return None
        return " ".join(texts)
