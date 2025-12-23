#!/usr/bin/env bash
# Build a single executable for the companion backend + caregiver GUI.
# Usage:
#   cd companion_core
#   bash build_app.sh
set -euo pipefail

APP_NAME="JacksonCompanion"
MAIN_SCRIPT="jackson_companion_v15_3.py"

# Create venv if missing
if [ ! -d ".venv" ]; then
  python3 -m venv .venv
fi
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install pyinstaller

pyinstaller \
  --onefile \
  --windowed \
  --name "${APP_NAME}" \
  "${MAIN_SCRIPT}"

echo "Build complete. Binary at dist/${APP_NAME}"
