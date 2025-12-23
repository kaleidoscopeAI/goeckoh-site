from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Literal, Optional
import random
import uuid
import time
import librosa
import numpy as np
import soundfile as sf
import python_speech_features as psf
from scipy.spatial.distance import cosine

