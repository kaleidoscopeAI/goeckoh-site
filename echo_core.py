"""
Echo v4.0 – Independent Crystalline Core
Implementation following the provided spec using only stdlib + numpy.
"""

import ctypes
from collections import deque
import json
import math
import os
import sys
import time

import numpy as np


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_CONFIG = {
    "sample_rate": 16000,
    "frame_size": 512,
    "hop_size": 256,
    "history_len": 32,
    "routine": {"contexts": 3, "features": 4, "alpha": 0.1},
    "seg": {"theta_rms": 0.02, "theta_zcr": 0.15, "min_voiced_run": 5},
    "text": {
        "n_fft": 1024,
        "win_length": 800,
        "hop_length": 200,
        "n_mels": 40,
        "epsilon": 1e-6,
    },
    "asr": {
        "hidden_size": 64,
        "blank_id": 0,
        "vocab": " _abcdefghijklmnopqrstuvwxyz'",
        "weight_path": "asr_weights.npz",
    },
    "tts": {
        "embedding_dim": 64,
        "decoder_dim": 64,
        "griffin_lim_iters": 16,
        "max_duration": 5,
        "weight_path": "tts_weights.npz",
    },
    "crystal": {
        "D": 24,
        "F": 14,
        "eta": 0.05,
        "tau_low": 0.4,
        "tau_high": 0.8,
        "gamma": 1.2,
    },
    "history_window": 8,
}


def load_config(path="config.json"):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        data = {}
    cfg = json.loads(json.dumps(DEFAULT_CONFIG))
    _deep_update(cfg, data)
    return cfg


def _deep_update(base, override):
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_update(base[k], v)
        else:
            base[k] = v


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------


def clamp(val, lo, hi):
    return max(lo, min(hi, val))


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def softmax(logits):
    logits = logits - np.max(logits)
    exp = np.exp(logits)
    denom = np.sum(exp)
    return exp / denom if denom != 0 else np.zeros_like(exp)


def ring_buffer_update(buffer, entry, limit):
    # Performance: Use deque with maxlen for O(1) append/pop instead of O(n) list.pop(0)
    # Convert list to deque if needed for backward compatibility
    if isinstance(buffer, list):
        buffer = deque(buffer, maxlen=limit)
    buffer.append(entry)
    return buffer


def now():
    return time.time()


def time_of_day_norm():
    return (now() % 86400.0) / 86400.0


def ensure_np(x, dtype=np.float32):
    arr = np.asarray(x, dtype=dtype)
    return arr.copy() if arr.flags["WRITEABLE"] is False else arr


# ---------------------------------------------------------------------------
# Audio I/O (ctypes shells with safe fallbacks)
# ---------------------------------------------------------------------------


class AudioInterface:
    """Minimal capture/playback shell. Falls back to zeros if OS hooks fail."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.mode = self._detect_mode()
        self.handle = None
        self._init_backend()

    def _detect_mode(self):
        if sys.platform.startswith("win"):
            return "windows"
        if sys.platform.startswith("linux"):
            return "linux"
        return "dummy"

    def _init_backend(self):
        try:
            if self.mode == "linux":
                self._init_alsa()
            elif self.mode == "windows":
                self._init_winmm()
        except Exception:
            self.mode = "dummy"
            self.handle = None

    # --- Windows (winmm) ---
    def _init_winmm(self):
        # Delay heavy setup until first read to keep the code simple and robust.
        self.winmm = ctypes.windll.winmm
        self.handle = ctypes.c_void_p()

    # --- Linux (ALSA) ---
    def _init_alsa(self):
        self.alsa = ctypes.cdll.LoadLibrary("libasound.so.2")
        self.handle = ctypes.c_void_p()
        self._alsa_configured = False

    def read_chunk(self):
        n_samples = self.cfg["frame_size"] * 4
        if self.mode == "windows":
            return self._read_winmm(n_samples)
        if self.mode == "linux":
            return self._read_alsa(n_samples)
        return np.zeros((n_samples,), dtype=np.float32)

    def _read_winmm(self, n_samples):
        # Placeholder: real winmm callbacks require a full event loop.
        return np.zeros((n_samples,), dtype=np.float32)

    def _read_alsa(self, n_samples):
        if not self._alsa_configured:
            # Minimal blocking capture setup; errors drop to dummy.
            SND_PCM_STREAM_CAPTURE = 1
            if self.alsa.snd_pcm_open(
                ctypes.byref(self.handle), b"default", SND_PCM_STREAM_CAPTURE, 0
            ) < 0:
                self.mode = "dummy"
                return np.zeros((n_samples,), dtype=np.float32)
            self._alsa_configured = True
        frame_count = n_samples
        buf = (ctypes.c_short * frame_count)()
        read = self.alsa.snd_pcm_readi(self.handle, buf, frame_count)
        if read < 0:
            return np.zeros((n_samples,), dtype=np.float32)
        data = np.frombuffer(buf, dtype=np.int16)[: read]
        return data.astype(np.float32) / 32768.0

    def play(self, samples):
        # No-op placeholder to keep within allowed libraries.
        return


# ---------------------------------------------------------------------------
# Haptics interface
# ---------------------------------------------------------------------------


class HapticInterface:
    def __init__(self, cfg):
        self.cfg = cfg
        self.mode = self._detect_mode()
        if self.mode == "windows":
            self._init_windows()

    def _detect_mode(self):
        if sys.platform.startswith("win"):
            return "windows"
        if sys.platform.startswith("linux"):
            return "linux"
        return "dummy"

    def _init_windows(self):
        class XINPUT_VIBRATION(ctypes.Structure):
            _fields_ = [
                ("wLeftMotorSpeed", ctypes.c_ushort),
                ("wRightMotorSpeed", ctypes.c_ushort),
            ]

        self.XINPUT_VIBRATION = XINPUT_VIBRATION
        try:
            self.xinput = ctypes.windll.xinput1_4
        except Exception:
            self.xinput = None
            self.mode = "dummy"

    def apply_pattern(self, pattern):
        if pattern is None:
            return
        if self.mode == "windows" and self.xinput:
            val = int(clamp(pattern.get("intensity", 0.0), 0.0, 1.0) * 65535)
            vib = self.XINPUT_VIBRATION(val, val)
            self.xinput.XInputSetState(0, ctypes.byref(vib))
        elif self.mode == "linux":
            self._apply_linux(pattern)
        # dummy mode intentionally ignores hardware

    def _apply_linux(self, pattern):
        # Minimal PWM/sysfs stub; adjust here for real hardware mapping.
        path = "/sys/class/pwm/pwmchip0/pwm0/duty_cycle"
        intensity = clamp(pattern.get("intensity", 0.0), 0.0, 1.0)
        nanos = int(intensity * 1_000_000_000)
        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write(str(nanos))
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Signal processing
# ---------------------------------------------------------------------------


def frame_audio(x, frame_size, hop_size):
    x = ensure_np(x, dtype=np.float32)
    if len(x) < frame_size:
        return np.zeros((0, frame_size), dtype=np.float32)
    num_frames = 1 + (len(x) - frame_size) // hop_size
    idx = np.tile(np.arange(frame_size), (num_frames, 1)) + np.arange(
        0, num_frames * hop_size, hop_size
    )[:, None]
    return x[idx]


def rms(frame):
    return math.sqrt(float(np.mean(frame * frame))) if frame.size else 0.0


def zcr(frame):
    if len(frame) < 2:
        return 0.0
    signs = np.sign(frame)
    return float(np.mean(signs[:-1] != signs[1:]))


def detect_voiced_runs(rms_vals, zcr_vals, theta_rms, theta_zcr, min_run):
    voiced = [
        1 if (r >= theta_rms and z <= theta_zcr) else 0
        for r, z in zip(rms_vals, zcr_vals)
    ]
    runs = []
    start = None
    for i, v in enumerate(voiced):
        if v == 1 and start is None:
            start = i
        if v == 0 and start is not None:
            if i - start >= min_run:
                runs.append((start, i - 1))
            start = None
    if start is not None and len(voiced) - start >= min_run:
        runs.append((start, len(voiced) - 1))
    if not runs:
        return None, voiced
    runs.sort(key=lambda r: r[1] - r[0], reverse=True)
    return runs[0], voiced


def slice_segment(x, run, frame_size, hop_size):
    start, end = run
    start_idx = start * hop_size
    end_idx = end * hop_size + frame_size
    return x[start_idx:end_idx]


def stft(signal, n_fft, hop_length, win_length):
    signal = ensure_np(signal)
    if len(signal) < win_length:
        pad = np.zeros((win_length - len(signal),), dtype=signal.dtype)
        signal = np.concatenate([signal, pad])
    window = np.hanning(win_length).astype(np.float32)
    frames = 1 + (len(signal) - win_length) // hop_length
    specs = []
    for i in range(frames):
        start = i * hop_length
        frame = signal[start : start + win_length] * window
        spec = np.fft.rfft(frame, n=n_fft)
        specs.append(spec)
    return np.stack(specs, axis=0)


def istft(spectrogram, n_fft, hop_length, win_length):
    window = np.hanning(win_length).astype(np.float32)
    num_frames = spectrogram.shape[0]
    out_len = hop_length * (num_frames - 1) + win_length
    y = np.zeros((out_len,), dtype=np.float32)
    win_sum = np.zeros((out_len,), dtype=np.float32)
    for i in range(num_frames):
        start = i * hop_length
        frame = np.fft.irfft(spectrogram[i], n=n_fft).astype(np.float32)
        y[start : start + win_length] += frame[:win_length] * window
        win_sum[start : start + win_length] += window * window
    nonzero = win_sum > 1e-8
    y[nonzero] /= win_sum[nonzero]
    return y


def mel_filterbank(n_fft, n_mels, sample_rate, fmin=0.0, fmax=None):
    fmax = fmax or (sample_rate / 2)
    def hz_to_mel(f):
        return 2595.0 * np.log10(1.0 + f / 700.0)

    def mel_to_hz(m):
        return 700.0 * (10 ** (m / 2595.0) - 1.0)

    mels = np.linspace(hz_to_mel(fmin), hz_to_mel(fmax), n_mels + 2)
    hz = mel_to_hz(mels)
    bins = np.floor((n_fft + 1) * hz / sample_rate).astype(int)
    fbanks = np.zeros((n_mels, n_fft // 2 + 1))
    for i in range(1, n_mels + 1):
        left, center, right = bins[i - 1], bins[i], bins[i + 1]
        if center == left:
            center += 1
        if right == center:
            right += 1
        for j in range(left, center):
            fbanks[i - 1, j] = (j - left) / (center - left)
        for j in range(center, right):
            fbanks[i - 1, j] = (right - j) / (right - center)
    return fbanks.astype(np.float32)


def mel_spectrogram(signal, mel_fb, cfg_text):
    S = stft(signal, cfg_text["n_fft"], cfg_text["hop_length"], cfg_text["win_length"])
    mag = np.abs(S)
    mel = np.dot(mag, mel_fb.T)
    mel = np.maximum(mel, cfg_text["epsilon"])
    return np.log(mel)


def griffin_lim(magnitude, cfg_text, n_iters):
    n_fft = cfg_text["n_fft"]
    hop = cfg_text["hop_length"]
    win = cfg_text["win_length"]
    angles = np.exp(2j * np.pi * np.random.rand(*magnitude.shape))
    spec = magnitude * angles
    for _ in range(n_iters):
        y = istft(spec, n_fft, hop, win)
        est = stft(y, n_fft, hop, win)
        angles = np.exp(1j * np.angle(est))
        spec = magnitude * angles
    return istft(spec, n_fft, hop, win)


# ---------------------------------------------------------------------------
# ASR
# ---------------------------------------------------------------------------


def init_asr_weights(F, H, V, seed=0):
    rng = np.random.default_rng(seed)
    return {
        "W1": rng.standard_normal((H, F)).astype(np.float32) * 0.1,
        "b1": np.zeros((H,), dtype=np.float32),
        "W_h": rng.standard_normal((H, H)).astype(np.float32) * 0.1,
        "W_x": rng.standard_normal((H, H)).astype(np.float32) * 0.1,
        "b_h": np.zeros((H,), dtype=np.float32),
        "W_o": rng.standard_normal((V, H)).astype(np.float32) * 0.1,
        "b_o": np.zeros((V,), dtype=np.float32),
    }


def load_asr_weights(path, F, H, V):
    if os.path.exists(path):
        data = dict(np.load(path))
        needed = {"W1", "b1", "W_h", "W_x", "b_h", "W_o", "b_o"}
        if needed.issubset(set(data.keys())):
            return {k: ensure_np(v, dtype=np.float32) for k, v in data.items()}
    return init_asr_weights(F, H, V)


def map_text_to_ids(text, vocab):
    lookup = {ch: i for i, ch in enumerate(vocab)}
    return [lookup.get(ch, 0) for ch in text.lower()]


def ids_to_text(ids, vocab, blank_id):
    tokens = []
    prev = None
    for idx in ids:
        if idx == prev:
            continue
        prev = idx
        if idx == blank_id:
            continue
        if idx < len(vocab):
            tokens.append(vocab[idx])
    return "".join(tokens)


def asr_decode(mel_spec, weights, cfg_asr):
    vocab = cfg_asr["vocab"]
    blank_id = cfg_asr["blank_id"]
    H = cfg_asr["hidden_size"]
    T, F = mel_spec.shape
    W1, b1 = weights["W1"], weights["b1"]
    W_h, W_x, b_h = weights["W_h"], weights["W_x"], weights["b_h"]
    W_o, b_o = weights["W_o"], weights["b_o"]
    h_rnn = np.zeros((H,), dtype=np.float32)
    decoded = []
    for t in range(T):
        x = mel_spec[t]
        h1 = np.tanh(np.dot(W1, x) + b1)
        h_rnn = np.tanh(np.dot(W_h, h_rnn) + np.dot(W_x, h1) + b_h)
        logits = np.dot(W_o, h_rnn) + b_o
        probs = softmax(logits)
        decoded.append(int(np.argmax(probs)))
    return ids_to_text(decoded, vocab, blank_id)


def first_person_projector(text):
    tokens = text.split(" ")
    result = []
    i = 0
    while i < len(tokens):
        cur = tokens[i]
        nxt = tokens[i + 1] if i + 1 < len(tokens) else None
        lower = cur.lower()
        if lower == "you" and nxt and nxt.lower() == "are":
            result.extend(["I", "am"])
            i += 2
            continue
        if lower == "your":
            result.append("my")
            i += 1
            continue
        if lower == "you":
            result.append("I")
            i += 1
            continue
        if result and result[-1] == "I" and cur.endswith("s") and len(cur) > 2:
            cur = cur[:-1]
        result.append(cur)
        i += 1
    return " ".join(result)


# ---------------------------------------------------------------------------
# Feature extraction and crystal update
# ---------------------------------------------------------------------------


def compute_features(seg, t_fp, state, cfg):
    Fs = cfg["sample_rate"]
    duration_sec = float(len(seg)) / Fs if seg is not None else 0.0
    if seg is None or len(seg) == 0:
        rms_g = peak = zcr_g = 0.0
    else:
        rms_g = math.sqrt(float(np.mean(seg * seg)))
        peak = float(np.max(np.abs(seg)))
        sign_changes = np.sum(np.sign(seg[:-1]) != np.sign(seg[1:]))
        zcr_g = float(sign_changes) / max(len(seg) - 1, 1)
    tokens = [tok for tok in t_fp.split(" ") if tok] if t_fp else []
    len_chars = len(t_fp)
    len_tokens = len(tokens)
    has_question = 1 if ("?" in t_fp) else 0
    negations = {"not", "can't", "dont", "don't", "no"}
    has_negation = 1 if any(tok.lower() in negations for tok in tokens) else 0
    repetition_ratio = 0.0
    if len_tokens > 0:
        counts = {}
        repeats = 0
        for tok in tokens:
            counts[tok] = counts.get(tok, 0) + 1
            if counts[tok] > 1:
                repeats += 1
        repetition_ratio = repeats / float(len_tokens)
    history = state.get("H", [])
    window = cfg.get("history_window", 8)
    last_entries = history[-window:] if window > 0 else history
    successes = [e.get("success", 0.0) for e in last_entries]
    success_rate = float(np.mean(successes)) if successes else 0.0
    m_values = [e.get("m", 0.0) for e in last_entries]
    last_m_mean = float(np.mean(m_values)) if m_values else 0.0
    last_speech_time = state.get("last_speech_time")
    time_since = now() - last_speech_time if last_speech_time else 1e3
    tod = time_of_day_norm()
    sin_time = math.sin(2 * math.pi * tod)
    cos_time = math.cos(2 * math.pi * tod)
    e_k = np.array(
        [
            rms_g,
            peak,
            duration_sec,
            zcr_g,
            len_chars,
            len_tokens,
            has_question,
            has_negation,
            repetition_ratio,
            success_rate,
            last_m_mean,
            time_since,
            sin_time,
            cos_time,
        ],
        dtype=np.float32,
    )
    derived = {
        "duration_sec": duration_sec,
        "success_rate": success_rate,
        "last_m_mean": last_m_mean,
        "time_since_last_speech": time_since,
    }
    return e_k, derived


def init_crystal_params(D, F, seed=0):
    rng = np.random.default_rng(seed)
    A = np.eye(D, dtype=np.float32) * 0.1
    B = rng.standard_normal((D, F)).astype(np.float32) * 0.05
    w_m = rng.standard_normal((D,)).astype(np.float32) * 0.1
    b_m = 0.0
    return A, B, w_m, b_m


def crystal_update(e_k, t_fp, state, cfg, A, B, w_m, b_m, derived):
    eta = cfg["crystal"]["eta"]
    q_prev = state.get("q")
    if q_prev is None or len(q_prev) != A.shape[0]:
        q_prev = np.zeros((A.shape[0],), dtype=np.float32)
    grad = np.dot(A, q_prev) - np.dot(B, e_k)
    q_k = q_prev - eta * grad
    m_k = float(sigmoid(np.dot(w_m, q_k) + b_m))
    entry = {
        "t_fp": t_fp,
        "m": m_k,
        "timestamp": now(),
        "mode": None,
        "duration": derived.get("duration_sec", 0.0),
        "success": 1.0 if t_fp else 0.0,
    }
    # Performance: Get or create deque for history buffer
    history = state.get("H")
    if history is None:
        history = deque(maxlen=cfg["history_len"])
    history = ring_buffer_update(history, entry, cfg["history_len"])
    R_prev = state.get("R")
    R_k = update_routine(R_prev, e_k, cfg)
    state_updated = {
        "q": q_k,
        "H": history,
        "R": R_k,
        "last_speech_time": now() if t_fp else state.get("last_speech_time"),
    }
    return state_updated, m_k


def update_routine(R_prev, e_k, cfg):
    contexts = cfg["routine"]["contexts"]
    features = cfg["routine"]["features"]
    alpha = cfg["routine"]["alpha"]
    if R_prev is None or R_prev.shape != (contexts, features):
        R_prev = np.zeros((contexts, features), dtype=np.float32)
    slice_e = e_k[:features]
    for c in range(contexts):
        R_prev[c] = (1.0 - alpha) * R_prev[c] + alpha * slice_e
    return R_prev


# ---------------------------------------------------------------------------
# Policy and minor grammar fix
# ---------------------------------------------------------------------------


def minor_grammar_fix(text):
    tokens = text.split(" ")
    fixed = []
    for tok in tokens:
        if fixed and fixed[-1] == "I" and tok.endswith("s") and len(tok) > 2:
            tok = tok[:-1]
        fixed.append(tok)
    return " ".join(fixed)


def policy(state, t_fp, m_k, cfg):
    tau_low = cfg["crystal"]["tau_low"]
    tau_high = cfg["crystal"]["tau_high"]
    if not t_fp:
        return "", "SILENT_SUPPORT", state
    if m_k <= tau_low:
        return t_fp, "PURE_ECHO", state
    if m_k <= tau_high:
        return minor_grammar_fix(t_fp), "SHAPED_ECHO", state
    return "", "SILENT_SUPPORT", state


# ---------------------------------------------------------------------------
# TTS / Haptics / GUI state
# ---------------------------------------------------------------------------


def init_tts_weights(vocab, emb_dim, dec_dim, n_mels, seed=0):
    rng = np.random.default_rng(seed)
    return {
        "E": rng.standard_normal((len(vocab), emb_dim)).astype(np.float32) * 0.1,
        "W_d": rng.standard_normal((emb_dim,)).astype(np.float32) * 0.05,
        "b_d": np.float32(0.0),
        "W_dec1": rng.standard_normal((emb_dim, dec_dim)).astype(np.float32) * 0.1,
        "b_dec1": np.zeros((dec_dim,), dtype=np.float32),
        "W_dec2": rng.standard_normal((dec_dim, n_mels)).astype(np.float32) * 0.1,
        "b_dec2": np.zeros((n_mels,), dtype=np.float32),
    }


def load_tts_weights(path, vocab, emb_dim, dec_dim, n_mels):
    if os.path.exists(path):
        data = dict(np.load(path))
        needed = {"E", "W_d", "b_d", "W_dec1", "b_dec1", "W_dec2", "b_dec2"}
        if needed.issubset(set(data.keys())):
            return {k: ensure_np(v, dtype=np.float32) for k, v in data.items()}
    return init_tts_weights(vocab, emb_dim, dec_dim, n_mels)


def map_text_tokens(text, vocab):
    lookup = {ch: i for i, ch in enumerate(vocab)}
    ids = [lookup.get(ch, 0) for ch in text.lower()]
    return np.array(ids, dtype=np.int64)


def tts_inference(text, weights, mel_fb, cfg, cfg_text):
    if not text:
        return np.zeros((0,), dtype=np.float32)
    ids = map_text_tokens(text, cfg["asr"]["vocab"])
    E = weights["E"]
    emb = E[ids]
    dur_logits = np.dot(emb, weights["W_d"]) + weights["b_d"]
    durations = np.maximum(
        1, np.rint(np.exp(np.minimum(dur_logits, cfg["tts"]["max_duration"])))
    ).astype(int)
    expanded = []
    for vec, dur in zip(emb, durations):
        for _ in range(int(dur)):
            expanded.append(vec)
    H_enc = np.array(expanded, dtype=np.float32)
    dec1 = np.tanh(np.dot(H_enc, weights["W_dec1"]) + weights["b_dec1"])
    mel = np.dot(dec1, weights["W_dec2"]) + weights["b_dec2"]
    mag = np.exp(mel)
    waveform = griffin_lim(mag, cfg_text, cfg["tts"]["griffin_lim_iters"])
    return waveform.astype(np.float32)


def haptic_pattern(m_k, mode, cfg):
    gamma = cfg["crystal"]["gamma"]
    intensity = clamp(gamma * m_k, 0.0, 1.0)
    if mode == "PURE_ECHO" and m_k <= cfg["crystal"]["tau_low"]:
        return {"type": "NONE", "intensity": 0.0, "values": np.zeros((1,))}
    if m_k <= cfg["crystal"]["tau_high"]:
        t = np.linspace(0, 1, 20)
        values = (np.sign(np.sin(2 * math.pi * 0.7 * t)) > 0).astype(np.float32)
        return {"type": "SOFT_PULSE", "intensity": intensity, "values": intensity * values}
    t = np.linspace(0, 1, 20)
    values = (np.sin(2 * math.pi * 0.3 * t) + 1.0) / 2.0
    return {"type": "DEEP_PURR", "intensity": intensity, "values": intensity * values}


def build_gui_state(state, t_fp, r_k, m_k, mode_k, derived):
    q = state.get("q", np.zeros((3,), dtype=np.float32))
    proj = [float(q[i]) if i < len(q) else 0.0 for i in range(3)]
    return {
        "timestamp": now(),
        "meltdown_risk": float(m_k),
        "mode": mode_k,
        "q_projection": proj,
        "last_child_text": t_fp,
        "last_echo_text": r_k,
        "duration_sec": derived.get("duration_sec", 0.0),
        "success_rate": derived.get("success_rate", 0.0),
        "last_m_mean": derived.get("last_m_mean", 0.0),
    }


# ---------------------------------------------------------------------------
# Core processing class
# ---------------------------------------------------------------------------


class EchoCore:
    def __init__(self, cfg=None):
        self.cfg = cfg or load_config()
        self.mel_fb = mel_filterbank(
            self.cfg["text"]["n_fft"],
            self.cfg["text"]["n_mels"],
            self.cfg["sample_rate"],
        )
        F_mel = self.cfg["text"]["n_mels"]
        H_asr = self.cfg["asr"]["hidden_size"]
        V = len(self.cfg["asr"]["vocab"])
        self.asr_weights = load_asr_weights(
            self.cfg["asr"]["weight_path"], F_mel, H_asr, V
        )
        self.tts_weights = load_tts_weights(
            self.cfg["tts"]["weight_path"],
            self.cfg["asr"]["vocab"],
            self.cfg["tts"]["embedding_dim"],
            self.cfg["tts"]["decoder_dim"],
            self.cfg["text"]["n_mels"],
        )
        self.A, self.B, self.w_m, self.b_m = init_crystal_params(
            self.cfg["crystal"]["D"], self.cfg["crystal"]["F"]
        )
        self.state = {
            "q": np.zeros((self.cfg["crystal"]["D"],), dtype=np.float32),
            # Performance: Use deque with maxlen for O(1) append operations
            "H": deque(maxlen=self.cfg["history_len"]),
            "R": np.zeros(
                (self.cfg["routine"]["contexts"], self.cfg["routine"]["features"]),
                dtype=np.float32,
            ),
            "last_speech_time": None,
        }
        self.audio = AudioInterface(self.cfg)
        self.haptics = HapticInterface(self.cfg)

    def step(self, raw_audio):
        frames = frame_audio(
            raw_audio, self.cfg["frame_size"], self.cfg["hop_size"]
        )
        seg, speech_flag, _ = self._segment(frames, raw_audio)
        t_raw, t_fp = self._text(seg, speech_flag)
        e_k, derived = compute_features(seg, t_fp, self.state, self.cfg)
        self.state, m_k = crystal_update(
            e_k, t_fp, self.state, self.cfg, self.A, self.B, self.w_m, self.b_m, derived
        )
        r_k, mode_k, _ = policy(self.state, t_fp, m_k, self.cfg)
        y_out = tts_inference(r_k, self.tts_weights, self.mel_fb, self.cfg, self.cfg["text"]) if mode_k != "SILENT_SUPPORT" else np.zeros((0,), dtype=np.float32)
        h_pattern = haptic_pattern(m_k, mode_k, self.cfg)
        self.haptics.apply_pattern(h_pattern)
        gui_state = build_gui_state(self.state, t_fp, r_k, m_k, mode_k, derived)
        return y_out, h_pattern, gui_state, {
            "t_raw": t_raw,
            "t_fp": t_fp,
            "r_k": r_k,
            "mode": mode_k,
            "m_k": m_k,
            "speech_flag": speech_flag,
        }

    def _segment(self, frames, raw_audio):
        cfg_seg = self.cfg["seg"]
        if len(frames) == 0:
            return None, 0, self.state
        rms_vals = [rms(f) for f in frames]
        zcr_vals = [zcr(f) for f in frames]
        run, voiced_flags = detect_voiced_runs(
            rms_vals,
            zcr_vals,
            cfg_seg["theta_rms"],
            cfg_seg["theta_zcr"],
            cfg_seg["min_voiced_run"],
        )
        if run is None:
            return None, 0, self.state
        seg = slice_segment(raw_audio, run, self.cfg["frame_size"], self.cfg["hop_size"])
        return ensure_np(seg), 1, self.state

    def _text(self, seg, speech_flag):
        if speech_flag == 0 or seg is None:
            return "", ""
        mel_spec = mel_spectrogram(seg, self.mel_fb, self.cfg["text"])
        t_raw = asr_decode(mel_spec, self.asr_weights, self.cfg["asr"])
        t_fp = first_person_projector(t_raw)
        return t_raw, t_fp


# ---------------------------------------------------------------------------
# Execution helper
# ---------------------------------------------------------------------------


def run_echo_core():
    cfg = load_config()
    core = EchoCore(cfg)
    while True:
        raw = core.audio.read_chunk()
        y_out, h_pat, gui_state, debug = core.step(raw)
        core.audio.play(y_out)
        # GUI integration should read gui_state via IPC/file; here we just sleep.
        time.sleep(0.01)


if __name__ == "__main__":
    try:
        run_echo_core()
    except KeyboardInterrupt:
        pass
