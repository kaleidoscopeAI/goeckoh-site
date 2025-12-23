 1 +#!/usr/bin/env python3
 2 +"""
 3 +Hardware loopback latency probe.
 4 +
 5 +Plays a short chirp via sounddevice, records from the default input, and
     estimates
 6 +round-trip latency by cross-correlation. Requires output→input loopback
    (cable or
 7 +mic near speaker). Keeps dependencies light: numpy + sounddevice.
 8 +"""
 9 +import numpy as np
10 +import sounddevice as sd
11 +from pathlib import Path
12 +
13 +DURATION = 0.5
14 +SAMPLE_RATE = 48000
15 +LEVEL = 0.2
16 +
17 +
18 +def main() -> None:
19 +    t = np.linspace(0, DURATION, int(SAMPLE_RATE * DURATION), endpoint=F
    alse)
20 +    chirp = LEVEL * np.sin(2 * np.pi * (500 + (4000 - 500) * t / DURATIO
    N) * t).astype(np.float32)
21 +
22 +    sd.default.samplerate = SAMPLE_RATE
23 +    sd.default.channels = 1
24 +
25 +    recording = sd.playrec(chirp, samplerate=SAMPLE_RATE, channels=1, bl
    ocking=True, input_dtype="float32")
26 +    rec = recording.flatten()
27 +
28 +    corr = np.correlate(rec, chirp, mode="full")
29 +    lag = int(np.argmax(corr) - (len(chirp) - 1))
30 +    latency_ms = lag * 1000.0 / SAMPLE_RATE
31 +
32 +    print(f"Estimated round-trip latency: {latency_ms:.1f} ms (lag sampl
    es: {lag})")
33 +    out_path = Path("backend/.run/loopback_recording.npy")
34 +    out_path.parent.mkdir(parents=True, exist_ok=True)
35 +    np.save(out_path, rec)
36 +    print(f"Saved recording to {out_path}")
37 +
38 +
39 +if __name__ == "__main__":
40 +    main()

