"""
Extracts F0 and RMS energy envelopes from a waveform.
"""
if wav.ndim > 1:
    wav = np.mean(wav, axis=1)
wav = np.asarray(wav, dtype=np.float32)

frame_length = max(int(sample_rate * frame_ms / 1000.0), 256)
hop_length = max(int(sample_rate * hop_ms / 1000.0), 128)

# 1. Pitch (F0) extraction using the YIN algorithm
# YIN is robust and commonly used for speech processing.
f0, voiced_flag, voiced_probs = librosa.pyin(
    y=wav,
    fmin=fmin_hz,
    fmax=fmax_hz,
    sr=sample_rate,
    frame_length=frame_length,
    hop_length=hop_length
)
# Fill NaNs in unvoiced frames with a reasonable value (e.g., median of voiced frames)
if np.any(voiced_flag):
    median_f0 = np.nanmedian(f0[voiced_flag])
    f0 = np.nan_to_num(f0, nan=median_f0)
else:
    f0.fill(150) # Fallback to a generic pitch if no voice is detected

# 2. Energy (RMS) extraction
rms = librosa.feature.rms(
    y=wav, frame_length=frame_length, hop_length=hop_length, center=True
)[0]

# 3. Time alignment
times = librosa.frames_to_time(
    np.arange(len(f0)), sr=sample_rate, hop_length=hop_length
)

# Ensure all outputs are clean float32 arrays
f0 = f0.astype(np.float32)
rms = np.maximum(rms, 1e-5).astype(np.float32) # Prevent log errors

return ProsodyProfile(
    f0_hz=f0,
    energy=rms,
    times_s=times.astype(np.float32),
    frame_length=frame_length,
    hop_length=hop_length,
    sample_rate=sample_rate
)-e 


