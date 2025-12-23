"""
Thin wrapper to apply prosody matching when librosa is available.
No-op when librosa/soundfile are missing.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

try:
    from goeckoh.voice.prosody import apply_prosody_to_tts, extract_prosody_from_int16, ProsodyFeatures
    PROSODY_AVAILABLE = True
except Exception:
    apply_prosody_to_tts = None  # type: ignore
    extract_prosody_from_int16 = None  # type: ignore
    ProsodyFeatures = None  # type: ignore
    PROSODY_AVAILABLE = False


def maybe_extract_prosody(audio_i16, sr: int):
    if not (PROSODY_AVAILABLE and audio_i16 is not None):
        return None
    try:
        return extract_prosody_from_int16(audio_i16, sr)
    except Exception:
        return None


def maybe_apply_prosody(tts_path: Path, target_features) -> None:
    if not (PROSODY_AVAILABLE and apply_prosody_to_tts and target_features):
        return
    try:
        apply_prosody_to_tts(tts_path, target_features)
    except Exception:
        return
