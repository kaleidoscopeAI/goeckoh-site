"""Small audio helpers that avoid heavy dependencies."""

from __future__ import annotations

from typing import Iterable, Generator, Optional

import numpy as np


def rms(audio: np.ndarray) -> float:
    """Root-mean-square energy for silence detection."""
    if audio is None or audio.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(audio))))


def chunked_audio(stream: Iterable[np.ndarray], seconds: float, samplerate: int) -> Generator[np.ndarray, None, None]:
    """Combine small chunks into fixed-length windows."""
    buffer: Optional[np.ndarray] = None
    chunk_frames = int(seconds * samplerate)
    for block in stream:
        buffer = block if buffer is None else np.concatenate([buffer, block], axis=0)
        while buffer is not None and len(buffer) >= chunk_frames:
            yield buffer[:chunk_frames]
            buffer = buffer[chunk_frames:] if len(buffer) > chunk_frames else None
