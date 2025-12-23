from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, time, timedelta
from pathlib import Path
from typing import Literal, Optional

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger

# Assuming these are available from the main system
from advanced_voice_mimic import VoiceCrystal, Style
from behavior_monitor import BehaviorMonitor

