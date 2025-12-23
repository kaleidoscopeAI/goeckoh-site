# config.py
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

# Where we store per-child profiles
PROFILES_DIR = BASE_DIR / "profiles"
PROFILES_DIR.mkdir(exist_ok=True, parents=True)

# Sample rate for all audio
SAMPLE_RATE = 16000

# OpenVoice_server HTTP endpoint
OPENVOICE_BASE_URL = "http://localhost:8000"

# Default label used when registering voice with OpenVoice_server
DEFAULT_VOICE_LABEL = "child_voice"

# WebSocket broadcast config for Voice Bubble
WS_HOST = "127.0.0.1"
WS_PORT = 8765

# Whisper model configuration
# Valid names: "tiny", "base", "small", "medium", "large"
WHISPER_MODEL_NAME = "tiny"   # safest default for CPU-only

# Licensing
LICENSE_FILE = BASE_DIR / "license.json"
LICENSE_PRODUCT_CODE = "GK"   # prefix, e.g. GK-ABCD-1234
