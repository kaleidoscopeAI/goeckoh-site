# ECHO_V4_UNIFIED/voice_timbre.py
# Use child noises/utterances to build an approximate voice sample
# that TTS can use as speaker_wav. We never play these noises directly,
# only use them as conditioning.
from __future__ import annotations
from pathlib import Path
from typing import Optional
import numpy as np
from scipy.io import wavfile
from config import CONFIG

AUTO_TIMBRE_NAME = "jackson_auto_timbre.wav"

def _auto_timbre_path() -> Path:
    """Returns the canonical path for the auto-generated timbre file."""
    return CONFIG.models.voice_samples_dir / AUTO_TIMBRE_NAME

def update_auto_timbre_from_audio(
    audio: np.ndarray,
    sample_rate: int,
) -> Path:
    """
    Overwrites the auto-timbre file with the latest child sound.
    This file is used as speaker_wav for TTS to approximate the child's voice.
    It converts float audio to int16 for standard WAV format compatibility.
    """
    CONFIG.models.voice_samples_dir.mkdir(parents=True, exist_ok=True)
    path = _auto_timbre_path()
    try:
        # Ensure audio is in a writable format (16-bit PCM)
        if np.issubdtype(audio.dtype, np.floating):
            audio_int16 = (audio * 32767).astype(np.int16)
        else:
            audio_int16 = audio.astype(np.int16)
        
        wavfile.write(path, sample_rate, audio_int16)
    except Exception as e:
        print(f"Error writing auto-timbre file at {path}: {e}")
    return path

def get_auto_timbre_wav() -> Optional[str]:
    """
    Returns the path to the current auto-timbre wav file if it exists, else None.
    """
    path = _auto_timbre_path()
    return str(path) if path.exists() else None
