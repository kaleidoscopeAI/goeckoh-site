from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict

import numpy as np

from core.settings import SystemSettings, load_settings
from speech_loop import SpeechLoop


