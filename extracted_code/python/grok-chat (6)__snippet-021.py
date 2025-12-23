Pythonfrom __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Literal, Optional

import numpy as np
import time

from .config import AudioSettings
from .voice_mimic import VoiceMimic
from .prosody import extract_prosody, apply_prosody_to_tts

