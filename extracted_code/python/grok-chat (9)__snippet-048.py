     6 +import json
     7 +import time
     8 +from pathlib import Path
     9 +import requests
    10 +
    11 +API = "http://localhost:8080/process"
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
    27 +def main():
    28 +    if not DATA_DIR.exists():
    29 +        print(f"Data dir missing: {DATA_DIR}")
    30 +        return
    31 +    files = sorted(GT.keys())
    32 +    ok = 0
    33 +    lat_sum = 0.0
    34 +    lat = []
    35 +    for fname in files:
    36 +        fpath = DATA_DIR / fname
    37 +        if not fpath.exists():
    38 +            print(f"Missing {fpath}")
    39 +            continue
    40 +        with fpath.open("rb") as f:
    41 +            payload = {"text": GT[fname]}  # send expected transcript as text for comparison
    42 +            t0 = time.time()
    43 +            # For this API we only pass text; to test ASR you'd need an /asr endpoint.
    44 +            # Here we use /process to check latency and echo.
    45 +            resp = requests.post(API, json=payload, timeout=10)
    46 +            dt = time.time() - t0
    47 +            lat.append(dt * 1000.0)
    48 +            if resp.status_code == 200:
    49 +                data = resp.json()
    50 +                echoed = data.get("result", "")
    51 +                if isinstance(echoed, dict):
    52 +                    echoed = echoed.get("text", "") or str(echoed)
    53 +                if str(GT[fname]).lower() in str(echoed).lower():
    54 +                    ok += 1
    55 +            else:
    56 +                print(f"{fname}: HTTP {resp.status_code}")
    57 +            lat_sum += dt
    58 +            print(f"{fname}: {dt*1000:.1f} ms")
    59 +
    60 +    n = len(files)
    61 +    if n:
    62 +        print(f"\nAccuracy (echo match): {ok}/{n}")
    63 +        print(f"Latency: p50={percentile(lat,50):.1f} ms p95={percentile(lat,95):.1f} ms")
    64 +    else:
    65 +        print("No files processed.")
    66 +
    67 +
    68 +def percentile(vals, q):
    69 +    if not vals:
    70 +        return 0.0
    71 +    vals = sorted(vals)
    72 +    k = (len(vals) - 1) * q / 100.0
    73 +    f = int(k)
    74 +    c = min(f + 1, len(vals) - 1)
    75 +    if f == c:
    76 +        return float(vals[f])
    77 +    return float(vals[f] + (vals[c] - vals[f]) * (k - f))
    78 +
    79 +
    80 +if __name__ == "__main__":
    81 +    main()

