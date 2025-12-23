import torch
import torchaudio
import numpy as np
import sounddevice as sd
import queue
import threading
from faster_whisper import WhisperModel
from TTS.api import TTS
import onnxruntime as ort
import os
import tempfile

