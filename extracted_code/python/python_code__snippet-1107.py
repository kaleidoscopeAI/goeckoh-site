def maybe_extract_prosody(audio_i16, sr: int):
    if not (PROSODY_AVAILABLE and audio_i16 is not None):
        return None
    try:
        return extract_prosody_from_int16(audio_i16, sr)
    except Exception:
        return None


def maybe_apply_prosody(tts_path: Path, target_features) -> None:
    if not (PROSODY_AVAILABLE and apply_prosody_to_tts and target_features):
        return
    try:
        apply_prosody_to_tts(tts_path, target_features)
    except Exception:
        return


