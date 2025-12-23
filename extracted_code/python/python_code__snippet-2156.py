import bio_audio
if hasattr(bio_audio, "BioAcousticEngine") or hasattr(bio_audio, "BioEngine"):
    RUST_AVAILABLE = True
