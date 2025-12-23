from __future__ import annotations

import json
import math
import os
import queue
import random
import re
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import librosa
import networkx as nx
import numpy as np
import pyttsx3
import sounddevice as sd
import torch
import whisper
from scipy.signal import butter, lfilter

