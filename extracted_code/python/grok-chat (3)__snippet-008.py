import torch
import numpy as np
import sounddevice as sd
import queue
import threading
import time
import tempfile
import os
import re
import librosa
import soundfile as sf
from pathlib import Path
import whisper
import pyttsx3
from scipy.signal import butter, lfilter

