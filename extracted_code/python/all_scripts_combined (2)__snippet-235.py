buf = io.BytesIO(data)
import wave

wf = wave.open(buf, "rb")
n_channels = wf.getnchannels()
sampwidth = wf.getsampwidth()
framerate = wf.getframerate()
frames = wf.readframes(wf.getnframes())
wf.close()
if sampwidth != 2:
    raise ValueError("expected 16-bit PCM")
audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
if n_channels > 1:
    audio = audio.reshape(-1, n_channels).mean(axis=1)
if framerate != 16_000:
    duration = len(audio) / framerate
    target_len = int(duration * 16_000)
    x_old = np.linspace(0.0, 1.0, num=len(audio), endpoint=False)
    x_new = np.linspace(0.0, 1.0, num=target_len, endpoint=False)
    audio = np.interp(x_new, x_old, audio)
return audio
