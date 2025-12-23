class ProsodyFeatures:
    f0_hz: np.ndarray       # shape [T]
    rms: np.ndarray         # shape [T]
    time_s: np.ndarray      # shape [T]
    f0_median: float
    rms_mean: float
    duration_s: float


def _extract_from_float(y: np.ndarray, sr: int) -> ProsodyFeatures:
    if y.ndim > 1:
        y = np.mean(y, axis=0)
    y = y.astype(np.float32)

    n_fft = 1024
    hop_length = 256

    f0 = librosa.yin(
        y,
        fmin=50.0,
        fmax=400.0,
        sr=sr,
        frame_length=n_fft,
        hop_length=hop_length,
    )
    rms = librosa.feature.rms(
        y=y,
        frame_length=n_fft,
        hop_length=hop_length,
    )[0]

    frames = np.arange(len(f0))
    time_s = frames * hop_length / float(sr)

    f0_valid = f0[np.isfinite(f0) & (f0 > 0)]
    if f0_valid.size == 0:
        f0_median = 0.0
    else:
        f0_median = float(np.median(f0_valid))

    rms_mean = float(np.mean(rms))
    duration_s = float(len(y) / float(sr))

    return ProsodyFeatures(
        f0_hz=f0,
        rms=rms,
        time_s=time_s,
        f0_median=f0_median,
        rms_mean=rms_mean,
        duration_s=duration_s,
    )


def extract_prosody_from_int16(audio_int16: np.ndarray, sr: int) -> ProsodyFeatures:
    y = audio_int16.astype(np.float32) / 32768.0
    return _extract_from_float(y, sr)


def _safe_ratio(a: float, b: float, default: float = 1.0) -> float:
    if b <= 0 or a <= 0:
        return default
    return a / b


def apply_prosody_to_tts(
    tts_wav_path: Path,
    target: ProsodyFeatures,
    pitch_limit_semitones: float = 5.0,
    tempo_limit_ratio: float = 0.5,  # max +/- 50% change
    loudness_limit_ratio: float = 2.0,
