// Cargo.toml dependencies
[dependencies]
# Audio I/O
cpal = "0.15"  # Cross-platform audio I/O
ringbuf = "0.3"  # Lock-free ring buffer
# ASR (STT)
whisper-rs = "0.7"  # Whisper.cpp bindings
ort = "2.0"  # ONNX Runtime for faster STT/VAD
# TTS & Voice Cloning
piper-rs = { git = "https://github.com/rhasspy/piper-rs" }  # Piper TTS
# or use: tts-rs = "0.1" for system TTS
# Concurrency
tokio = { version = "1.35", features = ["full"] }
crossbeam = "0.8"
# DSP
rustfft = "6.1"
realfft = "3.3"
# Communication
serde = { version = "1.0", features = ["derive"] }
bincode = "1.3"
