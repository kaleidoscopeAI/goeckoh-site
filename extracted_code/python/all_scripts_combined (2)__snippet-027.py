from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import whisper
from language_tool_python import LanguageTool

from broken_speech_tool import normalize_text

from .config import SpeechModelSettings


