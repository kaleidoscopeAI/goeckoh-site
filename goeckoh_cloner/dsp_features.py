from dataclasses import dataclass
from typing import Tuple

import numpy as np
import librosa
from scipy.signal import hilbert


@dataclass
class FrameFeatures:
    f0: np.ndarray
    zcr: np.ndarray
    energy: np.ndarray
    tilt: np.ndarray
    hnr: np.ndarray


def _estimate_hnr(y: np.ndarray, sr: int, frame_length: int, hop_length: int) -> np.ndarray:
    """Crude HNR estimate using harmonic vs noise energy via HPSS."""
    # librosa.decompose.hpss expects 2D array; we can hack via stft
    S = np.abs(librosa.stft(y, n_fft=frame_length, hop_length=hop_length)) ** 2
    H, P = librosa.decompose.hpss(S)
    h_energy = H.sum(axis=0)
    n_energy = P.sum(axis=0) + 1e-8
    return 10.0 * np.log10(h_energy / n_energy + 1e-8)


def extract_frame_features(
    y: np.ndarray,
    sr: int,
    frame_ms: float = 25.0,
    hop_ms: float = 10.0,
) -> FrameFeatures:
    """Extract frame-level features for the psychoacoustic engine."""
    if y.ndim > 1:
        y = np.mean(y, axis=0)

    frame_length = int(sr * frame_ms / 1000.0)
    hop_length = int(sr * hop_ms / 1000.0)

    # F0 with librosa.yin
    f0 = librosa.yin(
        y,
        fmin=60,
        fmax=600,
        frame_length=frame_length,
        hop_length=hop_length,
    )
    # Replace unvoiced (nan) with previous voiced or median
    if np.isnan(f0).any():
        voiced = np.where(~np.isnan(f0))[0]
        if len(voiced) == 0:
            f0[:] = 180.0
        else:
            median_f0 = np.median(f0[voiced])
            idx = np.arange(len(f0))
            for i in range(len(f0)):
                if np.isnan(f0[i]):
                    # nearest voiced neighbor
                    nearest = voiced[np.argmin(np.abs(voiced - i))]
                    f0[i] = f0[nearest] if not np.isnan(f0[nearest]) else median_f0

    # ZCR
    zcr = librosa.feature.zero_crossing_rate(
        y, frame_length=frame_length, hop_length=hop_length
    )[0]

    # Energy
    frames = librosa.util.frame(y, frame_length=frame_length, hop_length=hop_length)
    energy = np.sum(frames ** 2, axis=0)
    energy = energy / (np.max(energy) + 1e-8)

    # Spectral tilt: difference between low and high band energies
    S = np.abs(librosa.stft(y, n_fft=frame_length, hop_length=hop_length)) ** 2
    freqs = librosa.fft_frequencies(sr=sr, n_fft=frame_length)
    mid_idx = np.searchsorted(freqs, 2000.0)
    low_energy = S[:mid_idx, :].sum(axis=0)
    high_energy = S[mid_idx:, :].sum(axis=0) + 1e-8
    tilt = np.log10(low_energy / high_energy + 1e-8)

    # HNR
    hnr = _estimate_hnr(y, sr, frame_length, hop_length)

    # Align lengths (they can differ by 1)
    min_len = min(len(f0), len(zcr), len(energy), len(tilt), len(hnr))
    return FrameFeatures(
        f0=f0[:min_len],
        zcr=zcr[:min_len],
        energy=energy[:min_len],
        tilt=tilt[:min_len],
        hnr=hnr[:min_len],
    )
