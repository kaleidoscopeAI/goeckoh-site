#!/usr/bin/env bash
set -e

# Minimal offline setup for the speech companion.

# 1) Virtualenv
python3 -m venv .venv
source .venv/bin/activate

# 2) Core deps
pip install --upgrade pip
pip install -r requirements.txt

# 3) Real-Time-Voice-Cloning (open-source)
mkdir -p third_party
cd third_party

if [ ! -d "Real-Time-Voice-Cloning" ]; then
  echo "Cloning RTVC from GitHub (requires network once)..."
  git clone https://github.com/CorentinJ/Real-Time-Voice-Cloning.git
fi

cd Real-Time-Voice-Cloning
pip install -r requirements.txt
pip install -e .

# 4) Pre-download RTVC models (no API calls; static files)
# This triggers the built-in download routine; ignore audio errors in headless mode.
python demo_toolbox.py --no_sound 2>/dev/null || true

cd ..

echo
echo "Setup complete."
echo "Activate with: source .venv/bin/activate"
echo "Enroll a voice with: python main.py --enroll default"
echo "Run the loop with:   python main.py --loop"
