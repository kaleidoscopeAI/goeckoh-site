# ECHO_V4_SPEC_BEGIN
# ========================================
# Echo v4.0 – Independent Crystalline Core
# ========================================

import numpy as np
import json
import time
import os
import sys
import ctypes
import re
from collections import deque

# ========================================
# 0. HELPER FUNCTIONS & CLASSES
# ========================================

class RingBuffer(deque):
    def __init__(self, size):
        super().__init__(maxlen=size)
    def full(self):
        return len(self) == self.maxlen

# ========================================
# 1. M_AUDIO – KERNEL-LEVEL AUDIO ACQUISITION
# ========================================

class AudioInterface:
    """
    Handles audio I/O using ctypes to interface with ALSA on Linux.
    NOTE: This implementation is for Linux only.
    """
    _alsa = None
    
    # ALSA constants
    SND_PCM_STREAM_CAPTURE = 1
    SND_PCM_STREAM_PLAYBACK = 0
    SND_PCM_ACCESS_RW_INTERLEAVED = 3
    SND_PCM_FORMAT_S16_LE = 2 # 16-bit little-endian
    
    @staticmethod
    def _get_alsa_lib():
        if AudioInterface._alsa is None:
            try:
                AudioInterface._alsa = ctypes.cdll.LoadLibrary('libasound.so.2')
            except OSError:
                print("ERROR: libasound.so.2 not found. Please install ALSA development libraries (e.g., sudo apt-get install libasound2-dev)")
                sys.exit(1)
        return AudioInterface._alsa

    @staticmethod
    def open(cfg, is_capture=True):
        print(f"INFO: Opening ALSA {'capture' if is_capture else 'playback'} device...")
        alsa = AudioInterface._get_alsa_lib()
        
        device_name = b"default"
        stream_type = AudioInterface.SND_PCM_STREAM_CAPTURE if is_capture else AudioInterface.SND_PCM_STREAM_PLAYBACK
        
        handle_ptr = ctypes.c_void_p()
        err = alsa.snd_pcm_open(ctypes.byref(handle_ptr), device_name, stream_type, 0)
        if err < 0:
            print(f"ERROR: Cannot open audio device {device_name}: {alsa.snd_strerror(err).decode()}")
            return None

        # Hardware parameters
        hw_params_ptr = ctypes.c_void_p()
        alsa.snd_pcm_hw_params_malloc(ctypes.byref(hw_params_ptr))
        alsa.snd_pcm_hw_params_any(handle_ptr, hw_params_ptr)
        
        # Set params
        alsa.snd_pcm_hw_params_set_access(handle_ptr, hw_params_ptr, AudioInterface.SND_PCM_ACCESS_RW_INTERLEAVED)
        alsa.snd_pcm_hw_params_set_format(handle_ptr, hw_params_ptr, AudioInterface.SND_PCM_FORMAT_S16_LE)
        
        rate = ctypes.c_uint(cfg['audio']['sample_rate'])
        alsa.snd_pcm_hw_params_set_rate_near(handle_ptr, hw_params_ptr, ctypes.byref(rate), None)
        
        alsa.snd_pcm_hw_params_set_channels(handle_ptr, hw_params_ptr, 1, 0) # Mono
        
        # Apply params
        err = alsa.snd_pcm_hw_params(handle_ptr, hw_params_ptr)
        if err < 0:
            print(f"ERROR: Cannot set hardware parameters: {alsa.snd_strerror(err).decode()}")
            return None
            
        alsa.snd_pcm_hw_params_free(hw_params_ptr)
        
        alsa.snd_pcm_prepare(handle_ptr)
        
        print(f"INFO: ALSA device '{device_name.decode()}' opened successfully at {rate.value} Hz.")
        
        return {
            "cfg": cfg,
            "handle": handle_ptr,
            "is_capture": is_capture,
            "chunk_size": cfg['audio']['sample_rate'] # 1 second chunks
        }

    @staticmethod
    def read_chunk(mic_handle):
        if not mic_handle or not mic_handle['handle']:
            return np.zeros(mic_handle['cfg']['audio']['sample_rate'], dtype=np.float32)

        alsa = AudioInterface._get_alsa_lib()
        chunk_size = mic_handle['chunk_size']
        buffer = (ctypes.c_short * chunk_size)()
        
        frames_read = alsa.snd_pcm_readi(mic_handle['handle'], ctypes.byref(buffer), chunk_size)
        
        if frames_read < 0:
            # Attempt to recover from error (e.g., overrun)
            print(f"WARN: ALSA read error: {alsa.snd_strerror(frames_read).decode()}. Attempting recovery...")
            alsa.snd_pcm_prepare(mic_handle['handle'])
            return np.zeros(chunk_size, dtype=np.float32)

        # Convert C-style short array to numpy array
        np_buffer = np.ctypeslib.as_array(buffer, shape=(frames_read,))
        
        # Normalize to float32 in range [-1, 1]
        # int16 range is -32768 to 32767
        return np_buffer.astype(np.float32) / 32768.0

    @staticmethod
    def play(audio_data, playback_handle):
        if not playback_handle or not playback_handle['handle'] or len(audio_data) == 0:
            return

        alsa = AudioInterface._get_alsa_lib()
        
        # Convert float32 [-1, 1] to int16
        int_data = (audio_data * 32767).astype(np.int16)
        buffer = (ctypes.c_short * len(int_data)).from_buffer(int_data)
        
        frames_written = alsa.snd_pcm_writei(playback_handle['handle'], ctypes.byref(buffer), len(int_data))

        if frames_written < 0:
            print(f"WARN: ALSA write error: {alsa.snd_strerror(frames_written).decode()}. Attempting recovery...")
            alsa.snd_pcm_prepare(playback_handle['handle'])
        elif frames_written < len(int_data):
            print(f"WARN: Wrote only {frames_written}/{len(int_data)} frames.")

    @staticmethod
    def close(handle_obj):
        if handle_obj and handle_obj['handle']:
            alsa = AudioInterface._get_alsa_lib()
            alsa.snd_pcm_close(handle_obj['handle'])
            print(f"INFO: ALSA {'capture' if handle_obj['is_capture'] else 'playback'} device closed.")

def M_AUDIO(x_k, STATE_prev, cfg):
    """Module 1: Audio Framing"""
    print("M_AUDIO...")
    frame_size = cfg['audio']['frame_size']
    hop_size = cfg['audio']['hop_size']
    
    num_frames = (len(x_k) - frame_size) // hop_size + 1
    if num_frames <= 0:
        return np.array([]), STATE_prev

    frames = np.array([
        x_k[j*hop_size : j*hop_size + frame_size]
        for j in range(num_frames)
    ])
    return frames, STATE_prev

# ========================================
# 2. M_SEG – PURE SIGNAL PROCESSING SPEECH DETECTION
# ========================================

def M_SEG(x_k, frames_k, STATE_prev, cfg):
    """Module 2: Speech Segmentation using RMS and ZCR."""
    print("M_SEG...")
    if frames_k.shape[0] == 0:
        return None, 0, STATE_prev

    # Config values
    vad_cfg = cfg['vad']
    rms_threshold = vad_cfg['rms_threshold']
    zcr_threshold = vad_cfg['zcr_threshold']
    min_voiced_frames = vad_cfg['min_voiced_frames']
    frame_size = cfg['audio']['frame_size']
    hop_size = cfg['audio']['hop_size']

    # 1. Per-frame RMS
    rms = np.sqrt(np.mean(frames_k**2, axis=1))

    # 2. Per-frame ZCR
    # np.sign returns -1, 0, 1. We need to handle the zero case.
    # A small epsilon prevents zero values from causing issues with sign changes.
    zcr = np.mean(np.abs(np.diff(np.sign(frames_k + 1e-10), axis=1)), axis=1) / 2

    # 3. Voiced Frame Criterion
    is_voiced = (rms >= rms_threshold) & (zcr <= zcr_threshold)
    
    # 4. Segment Extraction
    voiced_indices = np.where(is_voiced)[0]
    if len(voiced_indices) < min_voiced_frames:
        return None, 0, STATE_prev

    # Find contiguous runs
    runs = []
    if len(voiced_indices) > 0:
        current_run = [voiced_indices[0]]
        for i in range(1, len(voiced_indices)):
            if voiced_indices[i] == voiced_indices[i-1] + 1:
                current_run.append(voiced_indices[i])
            else:
                runs.append(current_run)
                current_run = [voiced_indices[i]]
        runs.append(current_run)

    # Filter runs by minimum length
    valid_runs = [run for run in runs if len(run) >= min_voiced_frames]

    if not valid_runs:
        return None, 0, STATE_prev

    # 5. Choose the longest voiced run
    longest_run = max(valid_runs, key=len)
    j_start, j_end = longest_run[0], longest_run[-1]

    # 6. Construct the segment from the original audio chunk x_k
    start_sample = j_start * hop_size
    end_sample = (j_end * hop_size) + frame_size
    seg_k = x_k[start_sample:end_sample]

    speech_flag_k = 1
    
    print(f"INFO: Detected speech segment from frame {j_start} to {j_end} ({len(seg_k)} samples).")

    return seg_k, speech_flag_k, STATE_prev


# ========================================
# 3. M_TEXT – ASR + FIRST-PERSON PROJECTOR (P)
# ========================================

def _first_person_projector(text):
    """
    Projects a second-person ("you") string to a first-person ("I") string.
    Includes simple verb corrections.
    """
    tokens = text.split(" ")
    result = []
    i = 0
    while i < len(tokens):
        cur = tokens[i]
        nxt = tokens[i+1] if i+1 < len(tokens) else None

        if cur.lower() == "you" and nxt and nxt.lower() == "are":
            result.append("I")
            result.append("am")
            i += 2
            continue

        if cur.lower() == "your":
            result.append("my")
            i += 1
            continue

        if cur.lower() == "you":
            result.append("I")
            i += 1
            continue
            
        # Verb correction after "I"
        if len(result) > 0 and result[-1] == "I":
            # If verb is 3rd person singular like "runs", "wants"
            # This is a simplification. A proper check would involve a POS tagger.
            if cur.endswith("s") and len(cur) > 2 and not cur.endswith("ss"):
                cur = cur[:-1]

        result.append(cur)
        i += 1
    return " ".join(result)

def _mel_spectrogram(y, sr, n_fft, hop_length, win_length, n_mels):
    """NumPy-based Mel Spectrogram calculation."""
    # STFT
    stft_matrix = _stft(y, n_fft, hop_length, win_length)
    magnitude_spec = np.abs(stft_matrix)

    # Mel filterbank
    # This is a simplified version. A real one can be generated with librosa.filters.mel
    mel_basis = np.random.rand(n_mels, int(1 + n_fft // 2)) * 0.1 # Placeholder
    mel_spec = np.dot(mel_basis, magnitude_spec)
    
    # Log-Mel Spectrogram
    log_mel_spec = np.log(mel_spec + 1e-9)
    
    return log_mel_spec.T # Return as (Time, Freq)

def _greedy_ctc_decode(logits, vocab):
    """Greedy CTC decoder."""
    # Argmax at each time step
    indices = np.argmax(logits, axis=1)
    
    # Remove repeats
    raw_path = [indices[0]]
    for i in range(1, len(indices)):
        if indices[i] != indices[i-1]:
            raw_path.append(indices[i])
            
    # Remove blank symbol (assuming blank is the last token, index V-1)
    blank_idx = len(vocab)
    path = [p for p in raw_path if p != blank_idx]
    
    # Map to characters
    # Placeholder vocab
    return "".join([vocab.get(p, "?") for p in path])

def _asr_decode(seg_k, W_ASR, cfg):
    """ASR inference using NumPy."""
    print("INFO: Starting ASR decoding...")
    # 1. Mel Spectrogram
    asr_cfg = cfg['asr']
    mel_spec = _mel_spectrogram(
        seg_k,
        sr=cfg['audio']['sample_rate'],
        n_fft=asr_cfg['fft_size'],
        hop_length=asr_cfg['hop_size'],
        win_length=asr_cfg['window_size'],
        n_mels=asr_cfg['mel_filters']
    )
    
    T, F_mel = mel_spec.shape
    
    # Load weights
    W1 = W_ASR['W1']
    b1 = W_ASR['b1']
    Wh = W_ASR['Wh']
    Wx = W_ASR['Wx']
    bh = W_ASR['bh']
    Wo = W_ASR['Wo']
    bo = W_ASR['bo']

    # Simplified vocab for decoding
    # In a real model, this would be loaded from a file
    vocab = {i: c for i, c in enumerate("abcdefghijklmnopqrstuvwxyz '.,?")}


    # 2. Forward Pass
    h_rnn = np.zeros(asr_cfg['rnn_hidden_dim'])
    logits = np.zeros((T, asr_cfg['vocab_size']))

    for p in range(T):
        x_p = mel_spec[p, :]
        
        # Layer 1: Linear
        h1 = np.tanh(W1 @ x_p + b1)
        
        # Temporal RNN
        h_rnn = np.tanh(Wh @ h_rnn + Wx @ h1 + bh)
        
        # Output logits
        o_p = Wo @ h_rnn + bo
        logits[p, :] = o_p
        
    # Softmax (optional for greedy decode, but good practice)
    # exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    # probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

    # 3. Decode
    t_k = _greedy_ctc_decode(logits, vocab)
    print(f"INFO: ASR decoded text: '{t_k}'")
    
    return t_k

def M_TEXT(seg_k, speech_flag_k, STATE_prev, cfg, W_ASR):
    """Module 3: Transcription and Projection"""
    print("M_TEXT...")
    if speech_flag_k == 0 or seg_k is None:
        return "", "", STATE_prev
    
    # ASR model call
    t_k = _asr_decode(seg_k, W_ASR, cfg)
    
    # First-Person Projector P(t)
    t_fp_k = _first_person_projector(t_k)
    print(f"INFO: Raw text: '{t_k}', Projected text: '{t_fp_k}'")

    return t_k, t_fp_k, STATE_prev

# ========================================
# 4. M_FEAT & M_CRYSTAL – BEHAVIOR ENGINE
# ========================================

def M_FEAT(seg_k, t_fp_k, STATE_prev, cfg):
    """Module 4a: Feature Extraction"""
    print("M_FEAT...")
    
    # 1) Audio features
    if seg_k is not None and len(seg_k) > 0:
        sample_rate = cfg['audio']['sample_rate']
        duration_sec = len(seg_k) / sample_rate
        rms_global = np.sqrt(np.mean(seg_k**2))
        peak = np.max(np.abs(seg_k))
        zero_crossings_total = np.sum(np.abs(np.diff(np.sign(seg_k + 1e-10)))) / 2
        zcr_global = zero_crossings_total / (len(seg_k) - 1)
    else:
        duration_sec = 0
        rms_global = 0
        peak = 0
        zcr_global = 0

    # 2) Text features
    len_chars = len(t_fp_k)
    tokens = t_fp_k.split()
    len_tokens = len(tokens)
    has_question = 1 if "?" in t_fp_k else 0
    negation_words = {"not", "can't", "don't", "no", "n't"}
    has_negation = 1 if any(word in negation_words for word in tokens) else 0
    if len_tokens > 0:
        repetition_ratio = (len_tokens - len(set(tokens))) / len_tokens
    else:
        repetition_ratio = 0

    # 3) Historical features
    history = STATE_prev['H']
    S = 10 # Look at last 10 entries for historical stats
    if len(history) > 0:
        # A simple success metric could be related to low meltdown risk
        recent_history = list(history)[-S:]
        success_rate = np.mean([1 for entry in recent_history if entry.get('m', 1.0) < 0.5])
        last_m_mean = np.mean([entry.get('m', 0.0) for entry in recent_history])
        
        # Find last entry with speech to calculate time since last speech
        last_speech_entry = next((entry for entry in reversed(history) if entry.get('t_fp')), None)
        if last_speech_entry:
            time_since_last_speech = time.time() - last_speech_entry.get('timestamp', time.time())
        else:
            time_since_last_speech = 0 # No prior speech in history
    else:
        success_rate = 0
        last_m_mean = 0
        time_since_last_speech = 0
        
    # 4) Routine context
    # This would typically be based on a persistent routine model, here we simulate
    now = time.localtime()
    seconds_in_day = 24 * 60 * 60
    time_of_day = (now.tm_hour * 3600 + now.tm_min * 60 + now.tm_sec) / seconds_in_day
    sin_time = np.sin(2 * np.pi * time_of_day)
    cos_time = np.cos(2 * np.pi * time_of_day)
    
    e_k = np.array([
        rms_global,
        peak,
        duration_sec,
        zcr_global,
        len_chars,
        len_tokens,
        has_question,
        has_negation,
        repetition_ratio,
        success_rate,
        last_m_mean,
        time_since_last_speech,
        sin_time,
        cos_time
    ], dtype=np.float32)

    print(f"INFO: Feature vector e_k: {np.round(e_k, 2)}")
        
    return e_k, duration_sec, STATE_prev

def M_CRYSTAL(e_k, t_fp_k, duration_sec, STATE_prev, cfg, W_crystal):
    """Module 4b: Crystalline Core Update"""
    print("M_CRYSTAL...")
    q_prev = STATE_prev['q']
    
    # Load weights
    A = W_crystal['A']
    B = W_crystal['B']
    w_m = W_crystal['w_m']
    b_m = W_crystal['b_m']
    
    eta = cfg['crystal']['learning_rate']
    
    # Gradient descent step
    grad = A @ q_prev - B @ e_k
    q_k = q_prev - eta * grad
    
    # Meltdown risk
    z_m = w_m.T @ q_k + b_m
    m_k = 1 / (1 + np.exp(-z_m)) # Sigmoid

    # History update (mode is added later)
    entry = {
        "t_fp": t_fp_k, "m": float(m_k), "timestamp": time.time(),
        "duration": duration_sec, "mode": None
    }
    STATE_prev['H'].append(entry)

    STATE_k = {
        'q': q_k,
        'H': STATE_prev['H'],
        'R': STATE_prev['R'] # Routine matrix not implemented yet
    }
    
    return STATE_k, m_k


# ========================================
# 5. M_POLICY – RESPONSE MODE SELECTION
# ========================================

def M_POLICY(STATE_k, t_fp_k, m_k, cfg):
    """Module 5: Policy Engine"""
    print("M_POLICY...")
    tau_low = cfg['policy']['meltdown_threshold_low']
    tau_high = cfg['policy']['meltdown_threshold_high']
    
    if not t_fp_k:
        mode_k = "SILENT_SUPPORT"
        r_k = ""
    elif m_k <= tau_low:
        mode_k = "PURE_ECHO"
        r_k = t_fp_k
    elif m_k <= tau_high:
        mode_k = "SHAPED_ECHO"
        r_k = _first_person_projector(t_fp_k) # MINOR_GRAMMAR_FIX
    else:
        mode_k = "SILENT_SUPPORT"
        r_k = ""
        
    # Add mode to the latest history entry
    if len(STATE_k['H']) > 0:
        STATE_k['H'][-1]['mode'] = mode_k
        
    return r_k, mode_k, STATE_k

# ========================================
# 6. M_OUT – VOICE CLONING & HAPTICS
# ========================================

class HapticInterface:
    """
    Handles haptic feedback via sysfs PWM on Linux.
    Assumes a PWM device is available at /sys/class/pwm/pwmchip0/
    """
    PWM_CHIP_PATH = "/sys/class/pwm/pwmchip0"
    PWM_CHANNEL = 0
    PWM_PATH = f"{PWM_CHIP_PATH}/pwm0"
    # Set a period of 10ms (100 Hz), common for vibration motors
    PWM_PERIOD_NS = 10000000 

    @staticmethod
    def open(cfg):
        """Exports and configures the PWM channel."""
        print("INFO: Opening Haptic Interface (sysfs)...")
        if not os.path.isdir(HapticInterface.PWM_CHIP_PATH):
            print(f"WARN: PWM chip path not found at {HapticInterface.PWM_CHIP_PATH}. Haptics will be disabled.")
            return {"cfg": cfg, "enabled": False}

        # Export the PWM channel if not already exported
        if not os.path.isdir(HapticInterface.PWM_PATH):
            try:
                with open(f"{HapticInterface.PWM_CHIP_PATH}/export", "w") as f:
                    f.write(str(HapticInterface.PWM_CHANNEL))
                time.sleep(0.1) # Give sysfs time to create the directory
            except IOError as e:
                print(f"WARN: Failed to export PWM channel. Do you have permission? Error: {e}. Haptics disabled.")
                return {"cfg": cfg, "enabled": False}

        # Configure period and enable
        try:
            with open(f"{HapticInterface.PWM_PATH}/period", "w") as f:
                f.write(str(HapticInterface.PWM_PERIOD_NS))
            with open(f"{HapticInterface.PWM_PATH}/enable", "w") as f:
                f.write("1")
        except IOError as e:
            print(f"WARN: Failed to configure PWM channel. Error: {e}. Haptics disabled.")
            return {"cfg": cfg, "enabled": False}

        print("INFO: Haptic interface enabled.")
        return {"cfg": cfg, "enabled": True}

    @staticmethod
    def apply_pattern(h_vib, motor_handle):
        """Applies intensity by setting the duty cycle."""
        if not motor_handle or not motor_handle.get("enabled"):
            return

        # For now, we use the first value of the h_vib array as the intensity
        intensity = h_vib[0] if len(h_vib) > 0 else 0.0
        intensity = np.clip(intensity, 0, 1)
        
        duty_cycle_ns = int(intensity * HapticInterface.PWM_PERIOD_NS)
        
        try:
            with open(f"{HapticInterface.PWM_PATH}/duty_cycle", "w") as f:
                f.write(str(duty_cycle_ns))
        except IOError as e:
            print(f"WARN: Failed to set haptic duty cycle: {e}")

    @staticmethod
    def close(motor_handle):
        """Disables and unexports the PWM channel."""
        if not motor_handle or not motor_handle.get("enabled"):
            return
            
        print("INFO: Closing Haptic Interface...")
        try:
            # Disable motor first
            with open(f"{HapticInterface.PWM_PATH}/duty_cycle", "w") as f:
                f.write("0")
            with open(f"{HapticInterface.PWM_PATH}/enable", "w") as f:
                f.write("0")
            # Unexport
            with open(f"{HapticInterface.PWM_CHIP_PATH}/unexport", "w") as f:
                f.write(str(HapticInterface.PWM_CHANNEL))
        except IOError as e:
            print(f"WARN: Failed to cleanly close haptic interface: {e}")

def _stft(y, n_fft, hop_length, win_length):
    """NumPy-based Short-Time Fourier Transform."""
    # Hanning window
    window = np.hanning(win_length)
    
    # Pad the signal to ensure all frames are centered
    pad_len = (n_fft - win_length) // 2
    y = np.pad(y, pad_width=pad_len, mode='reflect')
    
    num_frames = (len(y) - n_fft) // hop_length + 1
    stft_matrix = np.empty((int(1 + n_fft // 2), num_frames), dtype=np.complex64)

    for i in range(num_frames):
        start = i * hop_length
        end = start + n_fft
        frame = y[start:end]
        windowed_frame = frame * window
        stft_matrix[:, i] = np.fft.rfft(windowed_frame, n_fft)
        
    return stft_matrix

def _istft(stft_matrix, hop_length, win_length):
    """NumPy-based Inverse Short-Time Fourier Transform."""
    n_fft = 2 * (stft_matrix.shape[0] - 1)
    num_frames = stft_matrix.shape[1]
    
    # Hanning window
    window = np.hanning(win_length)
    
    y = np.zeros(win_length + (num_frames - 1) * hop_length)
    
    for i in range(num_frames):
        start = i * hop_length
        end = start + win_length
        frame = np.fft.irfft(stft_matrix[:, i], n_fft)
        
        # Overlap-add
        y[start:end] += frame[:win_length] * window

    return y

def _griffin_lim(magnitude_spectrogram, n_fft, hop_length, win_length, n_iter):
    """Griffin-Lim algorithm for phase reconstruction."""
    print("INFO: Starting Griffin-Lim vocoder...")
    # Start with random phase
    angles = np.exp(2j * np.pi * np.random.rand(*magnitude_spectrogram.shape))
    
    for i in range(n_iter):
        # Full spectrogram
        stft_matrix = magnitude_spectrogram * angles
        # Inverse STFT
        y = _istft(stft_matrix, hop_length, win_length)
        # Re-calculate STFT
        stft_rebuilt = _stft(y, n_fft, hop_length, win_length)
        # Update phase
        angles = np.exp(1j * np.angle(stft_rebuilt))
        
    # Final ISTFT
    y_out = _istft(magnitude_spectrogram * angles, hop_length, win_length)
    print("INFO: Griffin-Lim finished.")
    return y_out.astype(np.float32)

def _layer_norm(x, eps=1e-5):
    mean = np.mean(x, axis=-1, keepdims=True)
    std = np.std(x, axis=-1, keepdims=True)
    return (x - mean) / (std + eps)

def _self_attention(x, W_Q, W_K, W_V):
    Q = x @ W_Q
    K = x @ W_K
    V = x @ W_V
    
    d_k = Q.shape[-1]
    attn_scores = (Q @ K.T) / np.sqrt(d_k)
    
    # Softmax
    exp_scores = np.exp(attn_scores - np.max(attn_scores, axis=-1, keepdims=True))
    attn_probs = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)
    
    return attn_probs @ V

def _feed_forward(x, W_ff1, b_ff1, W_ff2, b_ff2):
    # ReLU activation
    hidden = np.maximum(0, x @ W_ff1.T + b_ff1)
    return hidden @ W_ff2.T + b_ff2

def _tts_encoder_block(x, W_Q, W_K, W_V, W_ff1, b_ff1, W_ff2, b_ff2):
    attn_out = _self_attention(x, W_Q, W_K, W_V)
    x = _layer_norm(x + attn_out)
    ffn_out = _feed_forward(x, W_ff1, b_ff1, W_ff2, b_ff2)
    x = _layer_norm(x + ffn_out)
    return x

def _tts_decoder_block(x, W_Q, W_K, W_V, W_ff1, b_ff1, W_ff2, b_ff2):
    # Decoder is same as encoder in this simplified model
    return _tts_encoder_block(x, W_Q, W_K, W_V, W_ff1, b_ff1, W_ff2, b_ff2)

def _tts_inference(text, W_TTS, cfg):
    """TTS inference using NumPy (FastSpeech-like)."""
    print("INFO: Starting TTS inference...")
    tts_cfg = cfg['tts']
    
    # 1. Tokenizer (Placeholder)
    # In a real model, this would map text to a sequence of phoneme IDs
    phoneme_ids = np.random.randint(0, tts_cfg['phoneme_vocab_size'], size=len(text))

    # 2. Embedding
    H = W_TTS['E'][phoneme_ids]

    # 3. Encoder
    for i in range(tts_cfg['encoder_layers']):
        H = _tts_encoder_block(
            H,
            W_TTS[f'enc_{i}_W_Q'], W_TTS[f'enc_{i}_W_K'], W_TTS[f'enc_{i}_W_V'],
            W_TTS[f'enc_{i}_W_ff1'], W_TTS[f'enc_{i}_b_ff1'],
            W_TTS[f'enc_{i}_W_ff2'], W_TTS[f'enc_{i}_b_ff2']
        )
    H_enc = H

    # 4. Duration Predictor
    log_durations = H_enc @ W_TTS['dur_W'].T + W_TTS['dur_b']
    durations = np.floor(np.exp(log_durations)).astype(int).flatten()
    
    # 5. Length Regulator
    H_expanded = np.repeat(H_enc, durations, axis=0)
    
    # 6. Decoder
    H_dec = H_expanded
    for i in range(tts_cfg['decoder_layers']):
        H_dec = _tts_decoder_block(
            H_dec,
            W_TTS[f'dec_{i}_W_Q'], W_TTS[f'dec_{i}_W_K'], W_TTS[f'dec_{i}_W_V'],
            W_TTS[f'dec_{i}_W_ff1'], W_TTS[f'dec_{i}_b_ff1'],
            W_TTS[f'dec_{i}_W_ff2'], W_TTS[f'dec_{i}_b_ff2']
        )

    # 7. Final projection to Mel Spectrogram
    mel_spec = H_dec @ W_TTS['post_W'].T + W_TTS['post_b']
    
    print(f"INFO: Generated mel spectrogram of shape {mel_spec.T.shape}")
    return mel_spec.T

def M_OUT(STATE_k, r_k, t_fp_k, m_k, mode_k, cfg, W_TTS):
    """Module 6: Output Generation"""
    print("M_OUT...")
    
    # 6.1 TTS Voice Cloning
    if mode_k == "SILENT_SUPPORT" or not r_k:
        y_out_k = np.array([], dtype=np.float32)
    else:
        # Generate spectrogram from text
        mel_spec = _tts_inference(r_k, W_TTS, cfg)
        
        # Vocoder: Griffin-Lim
        y_out_k = _griffin_lim(
            np.exp(mel_spec), # Convert log-mel to magnitude
            n_fft=cfg['asr']['fft_size'],
            hop_length=cfg['asr']['hop_size'],
            win_length=cfg['asr']['window_size'],
            n_iter=cfg['tts']['gl_iterations']
        )

    # 6.2 Haptics
    gain = cfg['haptics']['gain']
    i_k = np.clip(gain * m_k, 0, 1)
    
    if mode_k == "PURE_ECHO" and m_k <= cfg['policy']['meltdown_threshold_low']:
        h_vib_k = np.array([0.0])
    else:
        h_vib_k = np.array([i_k]) # Simplified constant vibration

    # 6.3 GUI State
    history = STATE_k['H']
    success_rate = np.mean([1 for entry in history if entry['m'] < 0.5]) if history else 0
    last_m_mean = np.mean([entry['m'] for entry in history]) if history else 0
        
    GUI_state_k = {
        "timestamp": time.time(),
        "meltdown_risk": float(m_k),
        "mode": mode_k,
        "q_projection": STATE_k['q'][:3].tolist(),
        "last_child_text": t_fp_k,
        "last_echo_text": r_k,
        "duration_sec": history[-1]['duration'] if history else 0,
        "success_rate": success_rate,
        "last_m_mean": last_m_mean
    }

    return y_out_k, h_vib_k, GUI_state_k

def write_gui_state(state):
    # In a real scenario, this would use IPC
    with open("gui_state.json", "w") as f:
        json.dump(state, f, indent=4)

# ========================================
# 7. FINAL EXECUTION LOOP
# ========================================

def init_state(cfg):
    D = cfg['crystal']['state_dim']
    L = cfg['crystal']['history_len']
    return {
        'q': np.zeros(D, dtype=np.float32),
        'H': RingBuffer(size=L),
        'R': np.zeros((10, 10)) # Placeholder CxM routine matrix
    }

def run_echo_core():
    print("INFO: Starting Echo Core v4.0")
    # 1. Load config and weights
    with open("config.json", "r") as f:
        cfg = json.load(f)
    W_ASR = np.load("asr_weights.npz")
    W_TTS = np.load("tts_weights.npz")
    W_crystal = np.load("crystal_weights.npz")
    print("INFO: Configuration and all weights loaded.")

    # 2. Initialize interfaces
    mic = None
    playback = None
    motor = None
    try:
        mic = AudioInterface.open(cfg, is_capture=True)
        playback = AudioInterface.open(cfg, is_capture=False)
        motor = HapticInterface.open(cfg)
        if not mic or not playback or not motor:
            raise RuntimeError("Failed to initialize one or more interfaces.")
        print("INFO: Interfaces initialized.")

        # 3. Initialize state
        STATE = init_state(cfg)
        print("INFO: System state initialized.")

        while True:
            print("\n--- Cycle Start ---")
            # --- capture audio chunk via OS ---
            raw_audio = AudioInterface.read_chunk(mic)

            # --- M_AUDIO ---
            frames, STATE = M_AUDIO(raw_audio, STATE, cfg)

            # --- M_SEG ---
            seg, speech_flag, STATE = M_SEG(raw_audio, frames, STATE, cfg)

            # --- M_TEXT ---
            t_raw, t_fp, STATE = M_TEXT(seg, speech_flag, STATE, cfg, W_ASR)

            # --- M_FEAT ---
            e_k, duration, STATE = M_FEAT(seg, t_fp, STATE, cfg)

            # --- M_CRYSTAL ---
            STATE, m_k = M_CRYSTAL(e_k, t_fp, duration, STATE, cfg, W_crystal)

            # --- M_POLICY ---
            r_k, mode_k, STATE = M_POLICY(STATE, t_fp, m_k, cfg)

            # --- M_OUT ---
            y_out, h_vib, GUI_state = M_OUT(STATE, r_k, t_fp, m_k, mode_k, cfg, W_TTS)

            # --- Output ---
            AudioInterface.play(y_out, playback)
            HapticInterface.apply_pattern(h_vib, motor)
            write_gui_state(GUI_state)
            
            print(f"GUI State: meltdown_risk={GUI_state['meltdown_risk']:.2f}, mode={GUI_state['mode']}")
            
            # A short sleep to prevent busy-looping if read is non-blocking
            time.sleep(0.01) 
            
    except KeyboardInterrupt:
        print("\nINFO: Shutting down Echo Core.")
    except Exception as e:
        print(f"FATAL ERROR: An unexpected error occurred: {e}")
    finally:
        print("INFO: Cleaning up resources...")
        AudioInterface.close(mic)
        AudioInterface.close(playback)
        HapticInterface.close(motor)
        print("INFO: Cleanup complete.")
