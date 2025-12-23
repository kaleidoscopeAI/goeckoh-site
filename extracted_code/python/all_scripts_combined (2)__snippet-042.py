"""Audio similarity scoring using MFCC + DTW."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import librosa
import numpy as np
from fastdtw import fastdtw

from .config import AudioSettings


