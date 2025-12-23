from __future__ import annotations

import queue
from dataclasses import dataclass
from typing import Generator, Iterable, Optional

import numpy as np
import sounddevice as sd
import soundfile as sf
import librosa

from .config import AudioSettings


