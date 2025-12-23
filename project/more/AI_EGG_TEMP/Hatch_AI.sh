#!/bin/bash

# --- KALEIDOSCOPE AI HATCHING SCRIPT ---

echo "Initializing AI Egg... Please wait."

# Get the absolute path of the script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
BACKEND_DIR="$SCRIPT_DIR/system/backend"
FRONTEND_DIR="$SCRIPT_DIR/system/frontend/build"

# Detect OS and set portable Python path
if [[ "$(uname)" == "Darwin" ]]; then
    PYTHON_EXEC="$SCRIPT_DIR/system/portable_runtime/python_mac/bin/python3"
elif [[ "$(uname)" == "Linux" ]]; then
    PYTHON_EXEC="$SCRIPT_DIR/system/portable_runtime/python_linux/bin/python3"
else
    echo "ERROR: Unsupported OS. This script is for macOS and Linux."
    exit 1
fi

# --- Step 1: Launch AI Backend Server ---
echo "[1/3] Awakening cognitive core (Backend)..."
cd "$BACKEND_DIR"
# Activate the pre-installed virtual environment
source venv/bin/activate
# Start the Quart server in the background
"$PYTHON_EXEC" app.py &
BACKEND_PID=$!
echo "Backend process started with PID: $BACKEND_PID"
sleep 5 # Give the server a moment to start

# --- Step 2: Launch Visualization Server ---
echo "[2/3] Igniting visualization layer (Frontend)..."
cd "$FRONTEND_DIR"
# Use Python's simple HTTP server to serve the static frontend files
"$PYTHON_EXEC" -m http.server 8000 &
FRONTEND_PID=$!
echo "Frontend server started with PID: $FRONTEND_PID"
sleep 2

# --- Step 3: Open Browser and Hatch AI ---
echo "[3/3] Hatching complete. Opening interface in your browser..."
URL="http://localhost:8000"
# Cross-platform command to open URL
if [[ "$(uname)" == "Darwin" ]]; then
    open "$URL"
elif [[ "$(uname)" == "Linux" ]]; then
    xdg-open "$URL"
fi

echo ""
echo "✅ The AI is now live. Close this terminal to shut down the system."

# --- Cleanup on exit ---
trap 'kill $BACKEND_PID $FRONTEND_PID; echo "AI system shut down."; exit' INT TERM
wait
