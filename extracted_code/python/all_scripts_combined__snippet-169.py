from __future__ import annotations

import asyncio
import tempfile
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import AsyncGenerator, Iterable, Optional

import numpy as np

from audio.io import AudioIO
from audio.vad import VAD
from emotion.heart import EchoCrystallineHeart
from emotion.llm import LocalLLM
from core.logging import GuidanceLogger, MetricsLogger
from core.models import AttemptRecord, BehaviorEvent
from core.settings import SystemSettings, load_settings
from speech.stt import STT
from speech.text import TextProcessor
from voice.mimic import VoiceMimic
from voice.profile import VoiceProfile
from aba.engine import AbaEngine
from behavior_monitor import BehaviorMonitor
from loop.decision import AgentDecision, Mode
from loop.expressions import ExpressionGear, AudioData, Information, apply_prosody_to_tts


