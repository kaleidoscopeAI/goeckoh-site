from __future__ import annotations

import csv
import random
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

from core.models import BehaviorEvent
from core.settings import SystemSettings
from core.logging import GuidanceLogger
from voice.profile import VoiceProfile
from loop.decision import Mode
from aba.strategies import StrategyAdvisor, STRATEGIES, EVENT_TO_CATEGORIES


# ABA Skill Categories (from 2025 NIH/PMC overviews: self-care, communication, social/ToM)
