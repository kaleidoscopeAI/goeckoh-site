def _ensure_whisper() -> WhisperModel:
    """
    Lazy-load a small, CPU-friendly model.
    """
    global _WHISPER_MODEL
    if _WHISPER_MODEL is None:
        # tiny.en keeps CPU usage low; adjust if you need higher accuracy
        _WHISPER_MODEL = WhisperModel("tiny.en", device="cpu", compute_type="int8")
    return _WHISPER_MODEL


def transcribe_audio_block(audio: np.ndarray, samplerate: int = 16000) -> str:
    """
    Run local ASR on a raw mono float waveform.
    Returns lowercased transcript.
    """
    model = _ensure_whisper()
    segments, _ = model.transcribe(
        audio,
        language="en",
        beam_size=5,
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=600),
    )
    text = " ".join(seg.text for seg in segments).strip().lower()
    return text


def normalize_and_correct(text: str) -> str:
    s = text.strip()
    if not s:
        return ""
    s = s[0].upper() + s[1:]
    if not s.endswith((".", "!", "?")):
        s += "."
    return s


def dtw_similarity(a: str, b: str) -> float:
    ta = a.split()
    tb = b.split()
    if not ta or not tb:
        return 0.0
    vocab = {t: i for i, t in enumerate(sorted(set(ta + tb)))}
    va = np.array([vocab[t] for t in ta], dtype=float)
    vb = np.array([vocab[t] for t in tb], dtype=float)
    dist, _ = fastdtw(va, vb)
    max_len = max(len(ta), len(tb))
    return float(math.exp(-dist / max_len))


