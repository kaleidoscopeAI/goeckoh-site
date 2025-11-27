#!/bin/bash

#!/usr/bin/env bash
set -e

echo "Launching EchoVoice package..."

VENV_DIR=".venv"
if [ ! -d "$VENV_DIR" ]; then
    echo "Virtual environment not found at $VENV_DIR. Please run ./setup.sh first."
    exit 1
fi

source "$VENV_DIR/bin/activate"

echo "Using Python: $(python --version)"
echo "Starting EchoVoice CLI (run 'python -m echovoice --help' for options)..."
python -m echovoice "$@"
