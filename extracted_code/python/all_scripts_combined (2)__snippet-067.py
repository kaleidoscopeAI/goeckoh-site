from __future__ import annotations

import csv
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, List, Tuple
import numpy as np
import librosa

from flask import Flask, jsonify, render_template_string, request

from .advanced_voice_mimic import VoiceProfile, TTSEngine
from .settings_store import SettingsStore
import tempfile
import os
import soundfile as sf

from .config import CompanionConfig, CONFIG
from .calming_strategies import STRATEGIES


