#!/usr/bin/env bash
# This script sets up the environment, installs dependencies, builds, and runs the application.
set -euo pipefail

# 1. Setup Python environment
echo "Setting up Python virtual environment..."
if [ ! -d ".venv" ]; then
  python3 -m venv .venv
fi
source .venv/bin/activate

# 2. Install requirements
echo "Installing dependencies from requirements.txt..."
pip install --upgrade pip
pip uninstall -y torch torchaudio
pip install --index-url https://download.pytorch.org/whl/cpu torch torchaudio
pip install -r requirements.txt
pip install pyinstaller # Ensure pyinstaller is installed

# 3. Build the application
echo "Building the application..."
pyinstaller \
  --onefile \
  --windowed \
  --name "JacksonCompanion" \
  "jackson_companion_v15_3.py"

# 4. Run the application
echo "Running the application..."
./dist/JacksonCompanion
