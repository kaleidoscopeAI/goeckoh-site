import argparse
import io
import queue
import sys
import threading
import time
from pathlib import Path

import numpy as np
import sounddevice as sd
import soundfile as sf
import webrtcvad
import librosa

