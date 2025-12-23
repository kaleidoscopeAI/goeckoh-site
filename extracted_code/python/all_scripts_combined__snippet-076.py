from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional

import librosa
import numpy as np
import soundfile as sf

from core.settings import AudioSettings
from voice.mimic import VoiceMimic
from voice.profile import VoiceProfile
from voice.prosody import ProsodyProfile, extract_prosody
from loop.decision import AgentDecision, Mode


