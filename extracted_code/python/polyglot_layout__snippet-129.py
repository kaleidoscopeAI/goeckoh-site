  import wave, array
  x = np.asarray(samples, dtype=np.float32)
  x = np.clip(x, -1.0, 1.0)
  x = (x * 32767.0).astype(np.int16)
  with wave.open(str(path), "wb") as w:

