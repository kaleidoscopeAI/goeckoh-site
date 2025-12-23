"""Environment sanity checker for EchoVoice."""
from __future__ import annotations

import importlib
from pathlib import Path


REQUIRED = {
    "torch": "PyTorch for Heart lattice",
    "sounddevice": "Microphone capture/playback",
    "librosa": "Audio features/prosody",
    "scipy": "Audio snippet I/O",
    "faster_whisper": "STT",
    "pyttsx3": "Fallback TTS",
    "language_tool_python": "Grammar correction",
    "fastdtw": "Audio similarity",
}

OPTIONAL = {
    "TTS": "Coqui XTTS (real-time voice cloning)",
}


def check_import(name: str) -> bool:
    try:
        importlib.import_module(name)
        return True
    except Exception:
        return False


def main() -> None:
    missing = []
    optional_missing = []
    for mod, why in REQUIRED.items():
        ok = check_import(mod)
        status = "OK" if ok else "MISSING"
        print(f"{status:8} {mod:20} {why}")
        if not ok:
            missing.append(mod)

    for mod, why in OPTIONAL.items():
        ok = check_import(mod)
        status = "OK" if ok else "MISSING (optional)"
        print(f"{status:8} {mod:20} {why}")
        if not ok:
            optional_missing.append(mod)

    voice_sample = Path.home() / "EchoSystem" / "voices" / "child_voice.wav"
    print(f"Voice sample: {'FOUND' if voice_sample.exists() else 'MISSING'} at {voice_sample}")

    if missing:
        print("\nInstall missing modules (see project/requirements.pinned.txt).")
    if optional_missing:
        print("Optional (for real-time cloning): install TTS and provide a voice sample.")


if __name__ == "__main__":
    main()
