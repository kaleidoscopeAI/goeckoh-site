from __future__ import annotations

from dataclasses import dataclass, field

from core.paths import PathRegistry as Paths
from core.settings import (
    AudioSettings,
    BehaviorSettings,
    HeartSettings,
    SpeechSettings as SpeechModelSettings,
    SystemSettings,
    load_settings,
