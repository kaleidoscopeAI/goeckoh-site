import threading
import time
import numpy as np
import sounddevice as sd
from chatterbox import Chatterbox
from faster_whisper import WhisperModel
import language_tool_python
from scipy.signal import wiener
import logging
import re
import pyttsx3  # Fallback

