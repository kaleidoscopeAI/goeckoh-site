import time
import threading
import queue
import tempfile
import os
import re
from pathlib import Path
import numpy as np
import sounddevice as sd
import librosa
import soundfile as sf
import torch
import whisper
import pyttsx3
from scipy.signal import butter, lfilter
