from __future__ import annotations

from typing import Iterator

import numpy as np
import torch

from core.settings import AudioSettings


class VAD:
    """
    Voice Activity Detection using Silero VAD.

    This class is responsible for detecting speech in an audio stream.
    It uses the autism-tuned parameters from the documents.
    """

    def __init__(self, settings: AudioSettings):
        self.settings = settings
        self.model, self.utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad", model="silero_vad", force_reload=False
        )
        (
            self.get_speech_timestamps,
            self.save_audio,
            self.read_audio,
            self.VADIterator,
            self.collect_chunks,
        ) = self.utils

        self.vad_iterator = self.VADIterator(
            self.model,
            threshold=self.settings.vad_threshold,
            sampling_rate=self.settings.sample_rate,
            min_silence_duration_ms=self.settings.vad_min_silence_ms,
            speech_pad_ms=self.settings.vad_speech_pad_ms,
        )

    def process(self, audio_chunk: np.ndarray) -> dict | None:
        """
        Processes an audio chunk and returns speech timestamps.
        """
        return self.vad_iterator(audio_chunk, return_seconds=True)

    def reset(self) -> None:
        """Resets the VAD state."""
        self.vad_iterator.reset_states()

