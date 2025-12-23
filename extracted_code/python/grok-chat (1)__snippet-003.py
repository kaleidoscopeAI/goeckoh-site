import os
import numpy as np
import torch
import sounddevice as sd
import queue
import threading
import librosa
import pyttsx3  # Fallback TTS for mimicry simulation (adjust rate/pitch to mimic)
import logging
from dataclasses import dataclass
from typing import Optional

