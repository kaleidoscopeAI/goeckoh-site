from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

# ---------- AUDIO ----------
SAMPLE_RATE = 16000         # 16 kHz mono for ASR & VAD
CHANNELS = 1
FRAME_DURATION_MS = 30      # 30 ms frames for VAD
MAX_UTTERANCE_SECONDS = 12  # hard cap

# Silence detection: how long of continuous "non-speech" before we cut
VAD_SILENCE_MS = 800        # 0.8 seconds of silence
VAD_AGGRESSIVENESS = 2      # 0-3

# ---------- ASR ----------
WHISPER_MODEL_SIZE = "medium"   # can be "small", "medium", "large-v2"
WHISPER_DEVICE = "cpu"          # "cuda" if GPU
WHISPER_COMPUTE_TYPE = "int8"   # "float16" if GPU

# ---------- VOICE CLONE / RTVC ----------
RTVC_ROOT = BASE_DIR / "third_party" / "Real-Time-Voice-Cloning"
ENCODER_MODEL_FPATH = RTVC_ROOT / "encoder" / "saved_models" / "pretrained.pt"
SYNTHESIZER_MODEL_FPATH = RTVC_ROOT / "synthesizer" / "saved_models" / "pretrained" / "pretrained.pt"
VOCODER_MODEL_FPATH = RTVC_ROOT / "vocoder" / "saved_models" / "pretrained" / "pretrained.pt"

# Sample rate RTVC vocoder uses
RTVC_SAMPLE_RATE = 22050

# Directory where you store reference clips for the child
SPEAKER_REF_DIR = BASE_DIR / "speaker_refs"

# ---------- METRICS ----------
METRICS_DB_PATH = BASE_DIR / "metrics.jsonl"

# ---------- LATENCY ----------
MAX_ALLOWED_LATENCY_SEC = 3.0   # target: utterance -> playback within this
