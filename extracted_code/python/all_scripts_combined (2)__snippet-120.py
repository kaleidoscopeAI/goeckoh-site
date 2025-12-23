from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np
import soundfile as sf
import librosa
import random
from typing import Dict, List, Literal, Optional
import uuid

from .config import AudioSettings
from .voice import VoiceMimic
from .emotion import extract_prosody, ProsodyProfile
from .gears import AgentDecision, Information, AudioData

