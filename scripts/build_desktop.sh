#!/bin/bash

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

echo "=== EchoVoice Desktop Builder ==="

# 1. Virtual environment
if [ ! -d ".venv" ]; then
  echo "[1/4] No .venv found. Creating virtual environment..."
  python3 -m venv .venv
else
  echo "[1/4] Using existing .venv..."
fi

echo "Activating virtual environment..."
# shellcheck disable=SC1091
source .venv/bin/activate

# 2. Dependencies
echo "[2/4] Upgrading pip..."
pip install --upgrade pip

if [ -f "requirements.txt" ]; then
  echo "[2/4] Installing project dependencies from requirements.txt..."
  pip install -r requirements.txt
else
  echo "[2/4] WARNING: requirements.txt not found. Skipping bulk dependency install."
fi

echo "[2/4] Ensuring PyInstaller is installed..."
pip install pyinstaller

# 3. Spec validation
if [ ! -f "echovoice.spec" ]; then
  echo "[3/4] ERROR: echovoice.spec not found in project root: $PROJECT_ROOT"
  echo "Create or copy echovoice.spec into this directory and run again."
  exit 1
fi

echo "[3/4] Using spec file: echovoice.spec"

# 4. Build
echo "[4/4] Running PyInstaller..."
pyinstaller echovoice.spec

echo
echo "✅ Build complete."
echo "   Output folder: $PROJECT_ROOT/dist/echovoice"
echo "   On Windows: run dist/echovoice/echovoice.exe"
echo "   On macOS:   run dist/echovoice/echovoice.app (or open in Finder)"
echo "   On Linux:   run dist/echovoice/echovoice"

