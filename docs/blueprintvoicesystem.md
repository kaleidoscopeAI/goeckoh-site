"""
Goeckoh Real-Time Voice Pipeline
==================================

This script acts as the "glue" for the Goeckoh system, creating a single,
runnable pipeline that realizes the primary, non-negotiable goal:
a low-latency, offline, voice-to-clone feedback loop.

It wires together the best-of-breed components from the existing codebase
to perform the following sequence:
1. Capture a spoken utterance using an autism-optimized VAD.
2. Transcribe the utterance to text using a fast, offline STT engine.
3. Correct the text into a coherent, first-person narrative.
4. Synthesize the corrected text using a cloned voice.
5. Apply the prosody (pitch, rhythm, energy) of the original utterance
   to the synthesized audio.
6. Play the final, corrected, and prosody-matched audio back to the user.

This implementation prioritizes self-contained, low-latency components
to ensure real-time performance and offline privacy.

**To Run:**
1. Ensure all dependencies from `requirements.txt` and `requirements (2).txt` are installed.
2. Make sure you have run the enrollment process (`goeckoh_cloner/voice_logger.py`) to create a voice profile.
3. Run from the project root:
   `python realtime_voice_pipeline.py --profile-name <your_profile_name>`

"""

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

# --- Correcting Python Paths & Importing Components ---
# This script assumes it is run from the root of the 'bubble' project directory.
sys.path.insert(0, '.')

# 1. Audio Capture & VAD
from goeckoh_cloner.goeckoh_loop import AudioCapture, VADUtteranceDetector

# 2. STT (Speech-to-Text)
from src.neuro_backend import NeuroKernel # We will borrow the ASR loading logic

# 3. Text Correction
from goeckoh_cloner.correction_engine import clean_asr_text

# 4. Voice Cloning (TTS) & Profile Management
from tts import synthesize_text_to_wav # Using Coqui TTS implementation
from goeckoh_cloner.voice_profile import VoiceFingerprint, get_profile_path

# 5. Prosody Transfer
from goeckoh.voice.prosody import extract_prosody_from_int16, apply_prosody_to_tts

# --- Main Pipeline Orchestrator ---

class RealtimeVoicePipeline:
    def __init__(self, profile_name: str):
        print("✅ Initializing Goeckoh Real-Time Pipeline...")
        self.profile_name = profile_name
        self.stop_event = threading.Event()

        # Load Voice Profile
        self.profile_path = get_profile_path(self.profile_name)
        if not self.profile_path.exists():
            raise FileNotFoundError(f