"""Enrollment logger to build psychoacoustic fingerprints from real audio."""

import json
import os
from dataclasses import asdict
from typing import List

import numpy as np
from scipy.signal import find_peaks

from .attempt_analysis import analyze_attempt
from .voice_profile import SpeakerProfile, VoiceFingerprint


def log_voice_characteristics(
    audio_samples: List[np.ndarray],
    sr: int,
    user_id: str,
    output_dir: str,
    speaker_embedding: np.ndarray,
