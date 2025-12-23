from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Literal, Optional
import random
import uuid

import librosa
import numpy as np
import soundfile as sf

from .config import AudioSettings
from .prosody import ProsodyProfile, apply_prosody_to_tts, extract_prosody
from .voice_mimic import VoiceMimic

