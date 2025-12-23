  if x.dtype == np.int16:
     return (x.astype(np.float32) / 32768.0).clip(-1.0, 1.0)
  if x.dtype == np.int32:
     return (x.astype(np.float32) / 2147483648.0).clip(-1.0, 1.0)
  if x.dtype == np.uint8:
     return ((x.astype(np.float32) - 128.0) / 128.0).clip(-1.0, 1.0)
  return x.astype(np.float32).clip(-1.0, 1.0)

def mix_to_mono(x):
  return x if x.ndim == 1 else np.mean(x, axis=1)

def frame_signal(x, frame_length=2048, hop_length=512):
  n = x.shape[0]
  n_frames = 1 + int(np.ceil((n - frame_length) / hop_length)) if n >= frame_length else 1
  total_len = (n_frames - 1) * hop_length + frame_length
  pad = total_len - n
  if pad > 0:
     x = np.pad(x, (0, pad), mode="constant")
  strides = (x.strides[0]*hop_length, x.strides[0])
  frames = np.lib.stride_tricks.as_strided(x, shape=(n_frames, frame_length), strides=strides, writeable=False)
  return frames, pad

def rms_per_frame(x, frame_length=2048, hop_length=512):
  frames, _ = frame_signal(x, frame_length, hop_length)
  return np.sqrt(np.mean(frames**2, axis=1))

def zcr_per_frame(x, frame_length=2048, hop_length=512):
  frames, _ = frame_signal(x, frame_length, hop_length)
  signs = np.sign(frames)
  signs[signs == 0] = 1.0
  zc = np.sum(signs[:, 1:] * signs[:, :-1] < 0, axis=1)
  return zc

def power_spectrogram(x, sr, n_fft=2048, hop_length=512, window="hann"):
  from scipy.signal import get_window
  win = get_window(window, n_fft)
  f, t, Zxx = stft(x, fs=sr, window=win, nperseg=n_fft, noverlap=n_fft - hop_length, boundary=None, padded=True)
  S = np.abs(Zxx)**2
  return f, t, S

def spectral_centroid_bandwidth(f, S):
  eps = 1e-12
  mag = S + eps
  mag_sum = np.sum(mag, axis=0) + eps
  centroid = np.sum((f[:, None] * mag), axis=0) / mag_sum
  bw = np.sqrt(np.sum(((f[:, None] - centroid[None, :])**2) * mag, axis=0) / mag_sum)
  return centroid, bw

def spectral_rolloff(f, S, roll_percent=0.85):
  eps = 1e-12
  energy = S + eps
  cum = np.cumsum(energy, axis=0)
  total = cum[-1, :]
  targets = roll_percent * total
  idxs = np.argmax(cum >= targets[None, :], axis=0)
  return f[idxs]

def hz_to_mel(f):
  return 2595.0 * np.log10(1.0 + f / 700.0)

def mel_to_hz(m):
  return 700.0 * (10.0**(m / 2595.0) - 1.0)

def mel_filterbank(sr, n_fft=2048, n_mels=40, fmin=0.0, fmax=None):
  if fmax is None:
      fmax = sr / 2.0
  mel_min = hz_to_mel(fmin)
  mel_max = hz_to_mel(fmax)
  mels = np.linspace(mel_min, mel_max, n_mels + 2)
  freqs = mel_to_hz(mels)
  fft_freqs = np.linspace(0, sr / 2.0, n_fft // 2 + 1)
  fb = np.zeros((n_mels, len(fft_freqs)), dtype=np.float32)
  for i in range(1, n_mels + 1):
      f_left, f_center, f_right = freqs[i-1], freqs[i], freqs[i+1]
      left_idxs = np.where((fft_freqs >= f_left) & (fft_freqs <= f_center))[0]

