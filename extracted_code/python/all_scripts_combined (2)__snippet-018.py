    Extracts prosody (pitch and energy) from an audio signal.
    """
    f0, _, _ = librosa.pyin(
        y=audio,
        fmin=librosa.note_to_hz("C2"),
        fmax=librosa.note_to_hz("C7"),
        sr=sample_rate,
        frame_length=frame_length,
        hop_length=hop_length,
    )
    f0[~np.isfinite(f0)] = 0

    rms = librosa.feature.rms(
        y=audio, frame_length=frame_length, hop_length=hop_length
    )[0]

    return ProsodyProfile(
        f0_hz=f0,
        energy=rms,
        frame_length=frame_length,
        hop_length=hop_length,
        sample_rate=sample_rate,
    )
