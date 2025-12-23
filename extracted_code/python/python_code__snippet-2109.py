import bio_audio
if hasattr(bio_audio, "BioAcousticEngine"):
    RUST_AVAILABLE = True
    print("🦀 Rust bio-acoustic engine available")
else:
    print("⚠️  Rust engine module present but missing BioAcousticEngine symbol")
