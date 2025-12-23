 1 +#!/usr/bin/env python3
 2 +"""
 3 +Batch ASR check using the Free Spoken Digit mini-set.
 4 +Requires backend ASR endpoint at http://localhost:8080/asr.
 5 +"""
 6 +import time
 7 +import requests
 8 +from pathlib import Path
 9 +import io
10 +
11 +API = "http://localhost:8080/asr"
12 +DATA_DIR = Path(__file__).resolve().parents[1] / "backend" / "data" / "free-spoken-digit"
13 +
14 +GT = {
15 +    "0_jackson_0.wav": "zero",
16 +    "1_jackson_0.wav": "one",
17 +    "2_jackson_0.wav": "two",
18 +    "3_jackson_0.wav": "three",
19 +    "4_jackson_0.wav": "four",
20 +    "5_jackson_0.wav": "five",
21 +    "6_jackson_0.wav": "six",
22 +    "7_jackson_0.wav": "seven",
23 +    "8_jackson_0.wav": "eight",
24 +    "9_jackson_0.wav": "nine",
25 +}
26 +
27 +
28 +def main():
29 +    files = sorted(GT.keys())
30 +    if not DATA_DIR.exists():
31 +        print(f"Data dir missing: {DATA_DIR}")
32 +        return
33 +    ok = 0
34 +    lat = []
35 +    for fname in files:
36 +        fpath = DATA_DIR / fname
37 +        if not fpath.exists():
38 +            print(f"Missing {fpath}")
39 +            continue
40 +        with fpath.open("rb") as f:
41 +            audio_bytes = f.read()
42 +        t0 = time.time()
43 +        resp = requests.post(API, files={"file": (fname, io.BytesIO(audio_bytes), "audio/wav")}, timeout=15)
44 +        dt = (time.time() - t0) * 1000.0
45 +        lat.append(dt)
46 +        if resp.ok:
47 +            txt = resp.json().get("text", "").lower().strip()
48 +            if GT[fname] in txt:
49 +                ok += 1
50 +            print(f"{fname}: {dt:.1f} ms -> '{txt}'")
51 +        else:
52 +            print(f"{fname}: HTTP {resp.status_code} {resp.text}")
53 +    if lat:
54 +        print(f"\nAccuracy: {ok}/{len(files)}")
55 +        print(f"Latency: p50={percentile(lat,50):.1f} ms p95={percentile(lat,95):.1f} ms")
56 +
57 +
58 +def percentile(vals, q):
59 +    if not vals:
60 +        return 0.0
61 +    vals = sorted(vals)
62 +    k = (len(vals) - 1) * q / 100.0
63 +    f = int(k)
64 +    c = min(f + 1, len(vals) - 1)
65 +    if f == c:
66 +        return float(vals[f])
67 +    return float(vals[f] + (vals[c] - vals[f]) * (k - f))
68 +
69 +
70 +if __name__ == "__main__":
71 +    main()

