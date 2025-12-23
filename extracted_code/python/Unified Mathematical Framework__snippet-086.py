from __future__ import annotations
from pathlib import Path
from typing import Literal
import numpy as np
import soundfile as sf
import librosa
from TTS.api import TTS
import sounddevice as sd
import pyttsx3
import logging

