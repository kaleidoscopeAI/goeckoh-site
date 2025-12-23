import torch
import torch.nn as nn
import numpy as np
import sounddevice as sd
import queue
import threading
import tempfile
import os
import wave
import pyaudio
import math
import hashlib
import struct
import re
import json
import urllib.request
from dataclasses import dataclass
from typing import Optional, Dict, Any
from pathlib import Path
