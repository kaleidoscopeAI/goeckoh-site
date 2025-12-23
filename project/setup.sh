#!/usr/bin/env bash
set -e

echo "Setting up EchoVoice / Organic AI environment..."

VENV_DIR=".venv"
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment at $VENV_DIR..."
    python3 -m venv "$VENV_DIR"
fi

echo "Activating virtual environment..."
source "$VENV_DIR/bin/activate"

echo "Upgrading pip..."
python -m pip install --upgrade pip

echo "Installing Python dependencies from requirements.txt..."
python -m pip install -r requirements.txt

echo "Setup complete. Launch with ./run.sh (uses .venv and python -m echovoice)"
