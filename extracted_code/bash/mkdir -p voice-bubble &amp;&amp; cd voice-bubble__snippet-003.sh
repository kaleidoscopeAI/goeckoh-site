# Download models
wget https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-tiny.bin -O whisper-tiny.bin
wget https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US-lessac-medium.onnx -O voice.onnx

cargo run --release  # Single binary, 250MB total
