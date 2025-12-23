import pytest
from correction import correct_text
from tts import synthesize_text_to_wav
from pathlib import Path
import tempfile

def test_grammar_correction_basic():
    original = "this are a test sentence with bad grammar"
    corrected = correct_text(original)
    assert isinstance(corrected, str)
    # If language_tool_python is present, corrected should differ; otherwise unchanged is acceptable
    try:
        import language_tool_python  # noqa: F401
        assert corrected != original
    except Exception:
        assert corrected == original

def test_tts_requires_voice_profile(tmp_path):
    # Ensure synthesize_text_to_wav enforces a voice_profile when use_voice_clone is True
    with pytest.raises(RuntimeError):
        synthesize_text_to_wav("hello world", use_voice_clone=True, voice_profile_wav=None)

    # If a path is provided but not existing, expect an error
    fake_path = tmp_path / "fake.wav"
    with pytest.raises(RuntimeError):
        synthesize_text_to_wav("hello world", use_voice_clone=True, voice_profile_wav=str(fake_path))

# Optional: verify the get_wav_info helper if soundfile is installed
def test_get_wav_info(tmp_path):
    try:
        import soundfile  # noqa: F401
    except Exception:
        pytest.skip("soundfile not installed")
    # Create a small WAV file using soundfile so get_wav_info can read it
    import numpy as np
    import soundfile as sf
    wav_path = tmp_path / "test.wav"
    samplerate = 16000
    duration = 1.0
    t = np.linspace(0, duration, int(samplerate * duration), endpoint=False)
    data = 0.01 * np.sin(2*np.pi*440*t)
    sf.write(str(wav_path), data, samplerate)
    from speech_io import get_wav_info
    info = get_wav_info(str(wav_path))
    assert info["samplerate"] == samplerate
    assert abs(info["duration"] - duration) < 0.01
