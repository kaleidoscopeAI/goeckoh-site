1049              if sr != 16000:
1050 -                from scipy.signal import resample_poly  # type: ignore
1051 -                data = resample_poly(data, 16000, sr).astype(np.float32)
1050 +                # Lightweight resample to 16 kHz to avoid heavy SciPy dependency / ABI issues
1051 +                target_len = int(len(data) * 16000 / sr)
1052 +                idx = np.linspace(0, len(data), target_len, endpoint=False)
1053 +                data = np.interp(idx, np.arange(len(data)), data).astype(np.float32)
1054                  sr = 16000

