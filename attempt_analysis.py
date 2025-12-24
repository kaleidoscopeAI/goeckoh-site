"""
Psychoacoustic feature extraction for the Cloning Bubble system.
Extracts per-frame acoustic features using NumPy/SciPy only.

goeckoh/psychoacoustic_engine/attempt_analysis.py
"""

from dataclasses import dataclass
import numpy as np
from scipy import signal
from scipy.fft import rfft, rfftfreq


@dataclass
class AttemptFeatures:
    """Per-frame psychoacoustic features extracted from audio."""
    energy_attempt: np.ndarray       # [T] - RMS energy per frame
    f0_attempt: np.ndarray           # [T] - fundamental frequency (pitch)
    zcr_attempt: np.ndarray          # [T] - zero-crossing rate
    spectral_tilt: np.ndarray        # [T] - spectral slope (brightness)
    hnr_attempt: np.ndarray          # [T] - harmonics-to-noise ratio
    dt: float                        # seconds per frame


def _compute_rms_energy(y: np.ndarray, frame_length: int, hop_length: int) -> np.ndarray:
    """Compute RMS energy per frame."""
    n_frames = 1 + (len(y) - frame_length) // hop_length
    energy = np.zeros(n_frames)
    
    for i in range(n_frames):
        start = i * hop_length
        end = start + frame_length
        frame = y[start:end]
        energy[i] = np.sqrt(np.mean(frame ** 2))
    
    return energy


def _compute_zcr(y: np.ndarray, frame_length: int, hop_length: int) -> np.ndarray:
    """Compute zero-crossing rate per frame."""
    n_frames = 1 + (len(y) - frame_length) // hop_length
    zcr = np.zeros(n_frames)
    
    for i in range(n_frames):
        start = i * hop_length
        end = start + frame_length
        frame = y[start:end]
        signs = np.sign(frame)
        crossings = np.sum(np.abs(np.diff(signs))) / (2.0 * len(frame))
        zcr[i] = crossings
    
    return zcr


def _estimate_f0_autocorr(y: np.ndarray, sr: int, frame_length: int, hop_length: int) -> np.ndarray:
    """Estimate F0 using autocorrelation method."""
    n_frames = 1 + (len(y) - frame_length) // hop_length
    f0 = np.zeros(n_frames)
    
    min_lag = int(sr / 500)  # 500 Hz max
    max_lag = int(sr / 60)   # 60 Hz min
    
    for i in range(n_frames):
        start = i * hop_length
        end = start + frame_length
        frame = y[start:end]
        
        # Autocorrelation
        autocorr = np.correlate(frame, frame, mode='full')
        autocorr = autocorr[len(autocorr)//2:]
        
        # Find peak in valid range
        if max_lag < len(autocorr):
            search_region = autocorr[min_lag:max_lag]
            if len(search_region) > 0:
                peak_idx = np.argmax(search_region) + min_lag
                if autocorr[peak_idx] > 0.3 * autocorr[0]:  # Threshold
                    f0[i] = sr / peak_idx
    
    return f0


def _compute_spectral_tilt(y: np.ndarray, sr: int, frame_length: int, hop_length: int) -> np.ndarray:
    """Compute spectral tilt (slope of spectrum in dB)."""
    n_frames = 1 + (len(y) - frame_length) // hop_length
    tilt = np.zeros(n_frames)
    
    for i in range(n_frames):
        start = i * hop_length
        end = start + frame_length
        frame = y[start:end]
        
        # Apply window
        window = np.hanning(len(frame))
        frame_windowed = frame * window
        
        # FFT
        spectrum = np.abs(rfft(frame_windowed))
        freqs = rfftfreq(len(frame), 1/sr)
        
        # Convert to dB
        spectrum_db = 20 * np.log10(spectrum + 1e-10)
        
        # Linear regression on log-frequency vs dB
        if len(freqs) > 10:
            log_freqs = np.log10(freqs[1:] + 1)  # Skip DC
            coeffs = np.polyfit(log_freqs[:len(spectrum_db)-1], spectrum_db[1:], 1)
            tilt[i] = np.clip((coeffs[0] + 50) / 100, 0.0, 1.0)  # Normalize
    
    return tilt


def _compute_hnr(y: np.ndarray, sr: int, frame_length: int, hop_length: int) -> np.ndarray:
    """Estimate harmonics-to-noise ratio using autocorrelation."""
    n_frames = 1 + (len(y) - frame_length) // hop_length
    hnr = np.zeros(n_frames)
    
    for i in range(n_frames):
        start = i * hop_length
        end = start + frame_length
        frame = y[start:end]
        
        # Autocorrelation
        autocorr = np.correlate(frame, frame, mode='full')
        autocorr = autocorr[len(autocorr)//2:]
        
        # HNR from autocorrelation peak
        if len(autocorr) > 1:
            r0 = autocorr[0]
            r_max = np.max(autocorr[1:int(len(autocorr)/2)])
            if r0 > 0:
                hnr[i] = np.clip(r_max / r0, 0.0, 1.0)
    
    return hnr


def analyze_chunk(y: np.ndarray, sr: int, frame_length: int = 2048, hop_length: int = 512) -> AttemptFeatures:
    """
    Extract psychoacoustic features from audio using NumPy/SciPy.
    
    Args:
        y: Audio samples [N_samples]
        sr: Sample rate in Hz
        frame_length: Window size for analysis
        hop_length: Hop between frames
    
    Returns:
        AttemptFeatures with per-frame psychoacoustic data
    """
    if len(y) < frame_length:
        raise ValueError(f"Audio too short: {len(y)} samples < {frame_length}")
    
    y = y.astype(np.float32)
    
    # Extract features
    energy = _compute_rms_energy(y, frame_length, hop_length)
    f0 = _estimate_f0_autocorr(y, sr, frame_length, hop_length)
    zcr = _compute_zcr(y, frame_length, hop_length)
    tilt = _compute_spectral_tilt(y, sr, frame_length, hop_length)
    hnr = _compute_hnr(y, sr, frame_length, hop_length)
    
    dt = hop_length / sr
    
    return AttemptFeatures(
        energy_attempt=energy.astype(np.float32),
        f0_attempt=f0.astype(np.float32),
        zcr_attempt=zcr.astype(np.float32),
        spectral_tilt=tilt.astype(np.float32),
        hnr_attempt=hnr.astype(np.float32),
        dt=dt
    )
