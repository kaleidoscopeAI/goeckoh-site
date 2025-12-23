import bio_audio
# Prefer the modern BioAcousticEngine exported by rust_core
if hasattr(bio_audio, "BioAcousticEngine"):
    RUST_AVAILABLE = True
