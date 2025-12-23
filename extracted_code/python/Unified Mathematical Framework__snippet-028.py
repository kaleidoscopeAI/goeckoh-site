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
from faster_whisper import WhisperModel
from TTS.api import TTS
import torchaudio

