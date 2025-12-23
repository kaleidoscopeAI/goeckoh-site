class ProsodyProfile:
    f0_hz: np.ndarray
    energy: np.ndarray

def extract_prosody(wav: np.ndarray, sample_rate: int = 16_000, hop_ms: float = 10.0, f0_min: float = 75.0, f0_max: float = 600.0) -> ProsodyProfile:
    hop_length = int(sample_rate * (hop_ms / 1000.0))
    f0, voiced_flag, _ = librosa.pyin(wav, fmin=f0_min, fmax=f0_max, sr=sample_rate, frame_length=hop_length * 4, hop_length=hop_length)
    f0[np.isnan(f0)] = 0.0

    energy = librosa.feature.rms(y=wav, frame_length=hop_length * 4, hop_length=hop_length)[0]

    return ProsodyProfile(f0_hz=f0, energy=energy)

def apply_prosody_to_tts(tts_wav: np.ndarray, tts_sample_rate: int, prosody: ProsodyProfile, strength_pitch: float = 1.0, strength_energy: float = 1.0) -> np.ndarray:
    hop_length = len(tts_wav) // len(prosody.f0_hz) + 1
    frame_length = hop_length * 4
    num_frames = len(prosody.f0_hz)

    f0_tts, _, _ = librosa.pyin(tts_wav, fmin=75, fmax=600, sr=tts_sample_rate, frame_length=frame_length, hop_length=hop_length)

    voiced = ~np.isnan(f0_tts)
    if np.any(voiced):
        base_f0 = float(np.median(f0_tts[voiced]))
    else:
        base_f0 = float(np.median(prosody.f0_hz))

    out = np.zeros(len(tts_wav) + frame_length, dtype=np.float32)
    window = np.hanning(frame_length).astype(np.float32)
    eps = 1e-6

    for i in range(num_frames):
        start = i * hop_length
        end = start + frame_length
        if start >= len(tts_wav):
            break
        frame = tts_wav[start:end]
        if len(frame) < frame_length:
            frame = np.pad(frame, (0, frame_length - len(frame)), mode="constant")

        target_f0 = float(prosody.f0_hz[i])
        if base_f0 > 0:
            raw_ratio = target_f0 / base_f0
        else:
            raw_ratio = 1.0
        pitch_ratio = raw_ratio ** strength_pitch
        n_steps = 12.0 * np.log2(max(pitch_ratio, 1e-3))

        shifted = librosa.effects.pitch_shift(frame, sr=tts_sample_rate, n_steps=n_steps)

        if shifted.shape[0] != frame_length:
            if shifted.shape[0] > frame_length:
                shifted = shifted[:frame_length]
            else:
                shifted = np.pad(shifted, (0, frame_length - shifted.shape[0]))

        frame_rms = float(np.sqrt(np.mean(np.square(shifted)) + eps))
        target_rms = float(prosody.energy[i])
        if frame_rms > 0:
            ratio = (target_rms / frame_rms) ** strength_energy
        else:
            ratio = 1.0
        shifted *= ratio

        out[start:end] += shifted * window

    max_abs = float(np.max(np.abs(out)) + eps)
    if max_abs > 1.0:
        out = out / max_abs

    return out.astype(np.float32)
