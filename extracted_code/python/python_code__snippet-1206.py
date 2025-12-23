class AttemptFeatures:
    """All per-frame features needed by the Psychoacoustic Engine."""

    energy_attempt: np.ndarray  # [T] RMS energy
    f0_attempt: np.ndarray  # [T] F0 (Hz), NaNs handled
    zcr_attempt: np.ndarray  # [T] 0..1, Bouba/Kiki texture
    spectral_tilt: np.ndarray  # [T] 0..1, Metalness proxy
    hnr_attempt: np.ndarray  # [T] 0..1, Roughness proxy
    dt: float  # seconds per frame


def _frame_signal(y: np.ndarray, frame_length: int, hop_length: int) -> np.ndarray:
    """
    Return framed signal [T, frame_length] with padding as needed.
    This is the single source of truth for time axis T.
    """
    if y.ndim > 1:
        y = np.mean(y, axis=-1)
    y = y.astype(np.float32)

    if len(y) < frame_length:
        y = np.pad(y, (0, frame_length - len(y)))

    n_frames = 1 + max(0, (len(y) - frame_length) // hop_length)
    frames = np.zeros((n_frames, frame_length), dtype=np.float32)

    for i in range(n_frames):
        start = i * hop_length
        end = start + frame_length
        frame = y[start:end]
        if len(frame) < frame_length:
            frame = np.pad(frame, (0, frame_length - len(frame)))
        frames[i] = frame

    return frames


def _zero_crossing_rate(frames: np.ndarray) -> np.ndarray:
    T, L = frames.shape
    zcr = np.zeros(T, dtype=np.float32)
    for i in range(T):
        frame = frames[i]
        signs = np.sign(frame)
        zero_idx = np.where(signs == 0)[0]
        for idx in zero_idx:
            if idx == 0:
                signs[idx] = 1.0
            else:
                signs[idx] = signs[idx - 1]
        zcr[i] = np.sum(np.abs(np.diff(signs))) / (2.0 * (L - 1))
    return zcr


def _rms(frames: np.ndarray) -> np.ndarray:
    return np.sqrt(np.mean(frames**2, axis=1)).astype(np.float32)


def _simple_f0(frames: np.ndarray, sr: int, fmin: float = 50.0, fmax: float = 600.0) -> np.ndarray:
    """Autocorrelation-based F0 per frame."""
    T, L = frames.shape
    f0 = np.zeros(T, dtype=np.float32)
    for i in range(T):
        frame = frames[i]
        if not np.any(frame):
            continue
        autocorr = np.correlate(frame, frame, mode="full")[L - 1 :]
        max_val = np.max(autocorr)
        if max_val <= 0:
            continue
        autocorr = autocorr / (max_val + 1e-6)
        peaks, _ = find_peaks(autocorr, height=0.2)
        if peaks.size < 2:
            continue
        lags = np.diff(peaks)
        if lags.size == 0:
            continue
        avg_lag = float(np.median(lags))
        if avg_lag <= 0:
            continue
        f0_val = sr / avg_lag
        if fmin <= f0_val <= fmax:
            f0[i] = f0_val
    return f0


def _compute_psychoacoustic_features(frames: np.ndarray, sr: int) -> Dict[str, np.ndarray]:
    """
    Compute ZCR, spectral tilt, and crude HNR per frame using FFT.
    No external deps beyond numpy/scipy.
    """
    T, L = frames.shape

    # ZCR
    zcr = _zero_crossing_rate(frames)
    zcr = np.clip(zcr / (np.max(zcr) + 1e-6), 0.0, 1.0)

    # FFT
    fft = np.fft.rfft(frames, axis=1)  # [T, F]
    freqs = np.fft.rfftfreq(L, d=1.0 / sr)  # [F]
    power = np.abs(fft) ** 2 + 1e-12

    freqs_nz = freqs[1:]
    log_freqs = np.log10(np.maximum(freqs_nz, 1.0))  # [F-1]
    log_power = np.log10(power[:, 1:])  # [T, F-1]

    spectral_tilt = np.zeros(T, dtype=np.float32)
    hnr = np.zeros(T, dtype=np.float32)

    for t in range(T):
        p = log_power[t]
        if not np.any(np.isfinite(p)):
            spectral_tilt[t] = 0.5
            hnr[t] = 0.5
            continue

        coeffs = np.polyfit(log_freqs, p, 1)
        slope = float(coeffs[0])
        slope_clipped = np.clip(slope, -4.0, 0.0)
        spectral_tilt[t] = 1.0 - (abs(slope_clipped) / 4.0)

        spec = power[t]
        total_e = float(np.sum(spec))
        if total_e <= 0:
            hnr[t] = 0.5
        else:
            peak_e = float(np.max(spec))
            noise_e = max(total_e - peak_e, 1e-12)
            hnr[t] = peak_e / (peak_e + noise_e)

    return {
        "zcr": zcr.astype(np.float32),
        "spectral_tilt": spectral_tilt.astype(np.float32),
        "hnr": hnr.astype(np.float32),
    }


def analyze_attempt(
    y: np.ndarray,
    sr: int,
    frame_length: int = 2048,
    hop_length: int = 512,
