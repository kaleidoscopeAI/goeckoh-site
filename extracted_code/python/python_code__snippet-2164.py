fp = profile.fingerprint

phonemes = vocoder_backend.g2p(text)
if not phonemes:
    raise ValueError("No phonemes from G2P.")

child_duration = max(len(phonemes) / max(fp.rate, 1e-6), dt)
num_frames = max(1, int(child_duration / dt))
t = np.linspace(0.0, child_duration, num_frames)

base_contour = np.clip(np.sin(np.pi * t / child_duration), 0.0, 1.0)
target_f0 = fp.mu_f0 + base_contour * fp.sigma_f0
jitter_pert = np.sin(t * 100) * fp.jitter_base
target_f0 += jitter_pert * fp.sigma_f0 * 0.1

energy = np.clip(base_contour, 0.1, 1.0)
shimmer_pert = np.cos(t * 50) * fp.shimmer_base
energy += shimmer_pert * 0.1
energy = np.clip(energy, 0.1, 1.0)

target_hnr = np.full(num_frames, 1.0 - fp.base_roughness)
target_tilt = np.full(num_frames, fp.base_metalness)

zcr = np.zeros(num_frames)
frames_per_ph = max(1, num_frames // len(phonemes))
for i, ph in enumerate(phonemes):
    start = i * frames_per_ph
    end = num_frames if i == len(phonemes) - 1 else start + frames_per_ph
    zcr[start:end] = _phoneme_sharpness(ph, fp.base_sharpness)

audio = vocoder_backend.synthesize(
    phonemes, profile.embedding, target_f0, energy, target_hnr, target_tilt, dt
)

return audio, {
    "energy": energy.astype(np.float32),
    "f0": target_f0.astype(np.float32),
    "zcr": zcr.astype(np.float32),
    "hnr": target_hnr.astype(np.float32),
    "tilt": target_tilt.astype(np.float32),
    "dt": np.array([dt], dtype=np.float32),
}


