from __future__ import annotations

import csv
import json
import math
import os
import queue
import re
import threading
import time
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from socketserver import ThreadingMixIn
from typing import Any, Dict, List, Optional, Tuple

import librosa
import numpy as np
import requests
import sounddevice as sd
import soundfile as sf
import torch
from fastdtw import fastdtw
from scipy.signal import butter, lfilter
from TTS.api import TTS
from faster_whisper import WhisperModel  # local ASR

