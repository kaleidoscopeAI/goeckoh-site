from __future__ import annotations

import asyncio
import csv
import threading
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import tkinter as tk
from tkinter import ttk

from .config import CompanionConfig, CONFIG
from .agent import KQBCAgent, AGIStatus
from .speech_loop import SimulatedSpeechLoop


