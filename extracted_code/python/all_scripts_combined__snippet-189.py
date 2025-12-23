from __future__ import annotations

import random
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Literal, Optional

import librosa
import numpy as np
import soundfile as sf

from core.settings import AudioSettings, PathRegistry


