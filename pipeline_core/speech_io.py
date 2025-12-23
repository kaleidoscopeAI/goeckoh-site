import tempfile
from pathlib import Path

def _ensure_sound_deps():
    try:
        import sounddevice as sd  # noqa: F401
        import soundfile as sf  # noqa: F401
    except Exception as e:
        raise RuntimeError(
            "sounddevice and soundfile are required for recording/playback. "
            "Install: pip install sounddevice soundfile"
        ) from e

def record_to_wav(duration: float, out_path: str = None, samplerate: int = 16000):
    """Record from default microphone for 'duration' seconds and return saved WAV path."""
    _ensure_sound_deps()
    import sounddevice as sd
    import soundfile as sf
    import numpy as np

    if out_path is None:
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        out_path = tmp.name
        tmp.close()
    else:
        out_path = str(Path(out_path))
    channels = 1
    sd.default.samplerate = samplerate
    try:
        recording = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=channels, dtype="float32")
        sd.wait()
    except Exception as e:
        raise RuntimeError("Failed to record from microphone: " + str(e))
    # Normalize and save
    sf.write(out_path, recording, samplerate)
    return out_path

def playback_wav(path: str):
    """Play a WAV file via sounddevice."""
    _ensure_sound_deps()
    import sounddevice as sd
    import soundfile as sf

    data, samplerate = sf.read(path, dtype="float32")
    sd.play(data, samplerate)
    sd.wait()

def get_wav_info(path: str) -> dict:
    """
    Return basic info for a WAV file: {samplerate, duration, channels, frames}.
    Raises RuntimeError if soundfile is not available or file cannot be read.
    """
    _ensure_sound_deps()
    import soundfile as sf
    import numpy as np

    try:
        data, samplerate = sf.read(str(path), dtype="float32")
    except Exception as e:
        raise RuntimeError(f"Failed to read WAV file: {e}") from e
    frames = data.shape[0] if hasattr(data, "shape") else len(data)
    channels = data.shape[1] if data.ndim > 1 else 1
    duration = frames / float(samplerate)
    return {"samplerate": samplerate, "duration": duration, "channels": channels, "frames": frames}
