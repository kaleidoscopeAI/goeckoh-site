# 1. Install system dependencies (Linux)
sudo apt install portaudio19-dev ffmpeg

# 2. Install Python libraries
pip install -r requirements.txt

# 3. Ensure Ollama is running (The "Brain")
ollama pull deepseek-r1:8b
