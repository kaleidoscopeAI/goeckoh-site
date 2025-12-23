import torch
import torch.nn as nn
import numpy as np
import sounddevice as sd
import queue
import threading
import time
import tempfile
import os
import wave
import hashlib
import struct
import re
import json
import datetime
import csv
from dataclasses import dataclass
from typing import Optional, List, Dict
from pathlib import Path
import whisper
import pyttsx3
import librosa
import soundfile as sf
from scipy.spatial.distance import cosine
from threading import RLock

