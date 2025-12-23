from pathlib import Path
import tempfile

def synthesize_text_to_wav(text: str, out_wav: str = None, voice_profile_wav: str = None, use_voice_clone: bool = True):
    """
    Synthesizes 'text' to a WAV file using Coqui TTS + resemblyzer for voice cloning.
    This function enforces voice cloning: if 'use_voice_clone' is True, a valid voice_profile_wav
    must be provided and the TTS + resemblyzer libraries must be installed. There is no fallback TTS.
    Returns path to generated WAV.
    """
    out_wav = out_wav or (tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name)
    out_path = str(Path(out_wav))

    if use_voice_clone:
        if not voice_profile_wav:
            raise RuntimeError("Voice cloning is enabled but a 'voice_profile_wav' path was not provided.")
        # Try to import resemblyzer + TTS
        try:
            from resemblyzer import VoiceEncoder, preprocess_wav
            from TTS.api import TTS
        except Exception as e:
            raise RuntimeError(
                "Voice cloning requires 'resemblyzer' and 'TTS' packages. Install them via pip. "
                f"Original error: {e}"
            ) from e

        import os
        if not os.path.exists(voice_profile_wav):
            raise RuntimeError(f"Voice profile file does not exist: {voice_profile_wav}")

        try:
            encoder = VoiceEncoder()
            wav = preprocess_wav(voice_profile_wav)
            # Optional validation: embed = encoder.embed_utterance(wav)
        except Exception as e:
            raise RuntimeError("Failed to load or validate voice profile with resemblyzer: " + str(e)) from e

        try:
            # Prefer a multi-speaker model that supports speaker_wav
            tts = TTS("tts_models/en/vctk/vits")
            # The 'tts_to_file' method supports 'speaker_wav' for this model
            tts.tts_to_file(text=text, speaker_wav=voice_profile_wav, file_path=out_path)
            return out_path
        except Exception as e:
            raise RuntimeError("Voice cloning via TTS failed: " + str(e)) from e

    # This branch should not be used (we enforce cloning)
    raise RuntimeError("Voice cloning is enforced; 'use_voice_clone' must be True.")
