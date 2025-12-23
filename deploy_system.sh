#!/usr/bin/env bash
set -e
APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$APP_DIR/venv"
LOG_DIR="$APP_DIR/deployment_logs"
mkdir -p "$LOG_DIR"

GREEN='\033[1;32m'
RED='\033[1;31m'
NC='\033[0m'

pass_step() { echo -e "${GREEN}[OK] $1${NC}"; }
fail() { echo -e "${RED}[ERROR] $1${NC}"; exit 1; }

echo "=== ECHO V5.1 MASTER DEPLOYMENT ==="

# 1. SETUP ENVIRONMENT
if [ ! -d "$VENV_DIR" ]; then
    python3 -m venv "$VENV_DIR"
fi

"$VENV_DIR/bin/pip" install --upgrade pip wheel setuptools > /dev/null 2>&1
if ! "$VENV_DIR/bin/pip" install -r requirements.txt buildozer > "$LOG_DIR/install.log" 2>&1; then
    fail "Dependencies failed. Check deployment_logs/install.log"
fi
pass_step "Environment Ready"

# 2. ASSETS
STT="$APP_DIR/assets/model_stt"
TTS="$APP_DIR/assets/model_tts"
mkdir -p "$STT" "$TTS"

if [ ! -f "$STT/tokens.txt" ]; then
    echo " -> Downloading Neural Models..."
    wget -q -O stt.tar.bz2 https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-streaming-zipformer-en-2023-02-21.tar.bz2
    tar -xf stt.tar.bz2 -C "$APP_DIR/assets"
    mv "$APP_DIR/assets/sherpa-onnx-streaming-zipformer-en-2023-02-21/"* "$STT/"
    rm -rf "$APP_DIR/assets/sherpa-onnx-streaming-zipformer-en-2023-02-21/" stt.tar.bz2
fi

if [ ! -f "$TTS/en_US-lessac-medium.onnx" ]; then
    wget -q -O tts.tar.bz2 https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/vits-piper-en_US-lessac-medium.tar.bz2
    tar -xf tts.tar.bz2 -C "$APP_DIR/assets"
    mv "$APP_DIR/assets/vits-piper-en_US-lessac-medium/"* "$TTS/"
    rm -rf "$APP_DIR/assets/vits-piper-en_US-lessac-medium/" tts.tar.bz2
fi
pass_step "Assets Validated"

# 3. SYSTEM TESTS (Includes Dry Run)
echo "Running Pre-Flight Tests..."
export PYTHONPATH="$APP_DIR"
if "$VENV_DIR/bin/python" -m unittest discover -s tests -p "test_*.py" > "$LOG_DIR/tests.log" 2>&1; then
    pass_step "ALL TESTS PASSED (Logic + Pipeline Flow)"
else
    cat "$LOG_DIR/tests.log"
    fail "Critical Verification Failed"
fi

# 4. EXECUTION
ACTION=$1

if [ -z "$ACTION" ]; then
    echo "Select Target:"
    echo "1) Desktop: Clinician Dashboard"
    echo "2) Desktop: Child Interface"
    echo "3) Mobile: BUILD ANDROID APK"
    read -p "> " ACTION
fi


case $ACTION in
    1) "$VENV_DIR/bin/python" "src/main_app.py" --mode clinician ;;
    2) "$VENV_DIR/bin/python" "src/main_app.py" --mode child ;;
    3)
        echo "Building APK... (This takes time)"
        (
            export PATH="$VENV_DIR/bin:$PATH"
            "$VENV_DIR/bin/buildozer" android debug
        )
        ;;
esac
