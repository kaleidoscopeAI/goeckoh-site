def __init__(self, sr=16000):  # CREPE default 16kHz
    self.fs = sr
    self.chunk_size = 256
    self.rvc = None
    self.crepe = CREPE()  # Untrained; for exploration (load weights in prod from marl/crepe)

def extract_pitch_crepe(self, audio_data):
    """CREPE pitch detection (improved: frame-wise, weighted average to f0 Hz)."""
    audio = torch.from_numpy(audio_data).float().unsqueeze(0).unsqueeze(0)  # Batch, channel
    if audio.shape[-1] != 1024:
        logging.warning("Padding/resizing to 1024 samples for CREPE.")
        audio = nn.functional.pad(audio, (0, 1024 - audio.shape[-1]))
    with torch.no_grad():
        bins = self.crepe(audio)  # (1, 360)
    bins = bins.squeeze().numpy()
    cents = np.arange(360) * 20 - 199 * 20  # From C1 to B7, 20-cent bins
    weights = bins / np.sum(bins + 1e-8)
    cents_pred = np.sum(weights * cents)
    f0 = 10 * 2 ** (cents_pred / 1200)  # Ref 10 Hz
    logging.info(f"CREPE f0: {f0:.2f} Hz")
    return f0

def extract_params_from_audio(self, audio_data):
    """Improved with CREPE pitch (fallback to YIN)."""
    try:
        f0_avg = self.extract_pitch_crepe(audio_data)
    except Exception as e:
        logging.warning(f"CREPE failed ({e}); fallback to YIN.")
        waveform = torch.from_numpy(audio_data).float().unsqueeze(0)
        f0 = torchaudio.functional.detect_pitch_frequency(waveform, self.fs)
        f0_avg = torch.mean(f0[f0 > 0]).item()
    # Formants as previous
    formants = []  # LPC
    frame_len = 1024
    for i in range(0, len(audio_data) - frame_len, frame_len // 2):
        frame = audio_data[i:i+frame_len]
        if np.max(np.abs(frame)) < 1e-4: continue
        a = signal.lpc(frame, order=4)
        roots = np.roots(a)
        roots = roots[np.imag(roots) >= 0]
        freqs = np.arctan2(np.imag(roots), np.real(roots)) * (self.fs / (2 * np.pi))
        freqs = sorted(freqs[freqs > 90])[:4]
        formants.append(freqs)
    avg_formants = np.mean(formants, axis=0) if formants else [500, 1500, 2500, 3500]
    tract_scale = np.mean(avg_formants) / np.mean([500, 1500, 2500, 3500])
    return {'f0_avg': f0_avg, 'tract_scale': tract_scale}

def compare_voices(self, f0_pred, f0_gt, voiced_pred, voiced_gt, emb_pred, emb_gt, threshold=0.2):
    """Voice similarity comparisons (improved: GPE, VDE, FFE from paper; cosine embedding)."""
    # GPE: % voiced frames with rel err > threshold
    voiced = (f0_gt > 0) & (f0_pred > 0) & (voiced_gt & voiced_pred)
    if np.sum(voiced) == 0:
        gpe = 0.0
    else:
        rel_err = np.abs(f0_pred[voiced] - f0_gt[voiced]) / f0_gt[voiced]
        gpe = np.mean(rel_err > threshold)

    # VDE: % mismatched voicing
    vde = np.mean(voiced_pred != voiced_gt)

    # FFE: Avg GPE + VDE
    ffe = gpe + vde

    # Embedding cosine similarity
    cos_sim = np.dot(emb_pred, emb_gt) / (np.linalg.norm(emb_pred) * np.linalg.norm(emb_gt))

    logging.info(f"GPE: {gpe:.2%}, VDE: {vde:.2%}, FFE: {ffe:.2%}, Cosine Sim: {cos_sim:.4f}")
    return {'gpe': gpe, 'vde': vde, 'ffe': ffe, 'cos_sim': cos_sim}

# ... (rest as previous: xtts, rvc, etc.)

