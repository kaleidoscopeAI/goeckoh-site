     7 +import json
     8 +import math
     9 +import os
    10 +from pathlib import Path
    11 +import statistics
    12 +
    13 +import numpy as np
    14 +
    15 +try:
    16 +    import matplotlib.pyplot as plt
    17 +    HAVE_PLOT = True
    18 +except Exception:
    19 +    HAVE_PLOT = False
    20 +
    21 +DEFAULT_PATH = Path(__file__).resolve().parents[1] / "backend" / ".run"
        / "mirror_metrics.jsonl"
    22 +
    23 +def percentile(vals, q):
    24 +    if not vals:
    25 +        return math.nan
    26 +    vals = sorted(vals)
    27 +    k = (len(vals) - 1) * q / 100.0
    28 +    f = math.floor(k)
    29 +    c = min(f + 1, len(vals) - 1)
    30 +    if f == c:
    31 +        return float(vals[int(k)])
    32 +    return float(vals[f] + (vals[c] - vals[f]) * (k - f))
    33 +
    34 +def main(path: Path):
    35 +    if not path.exists():
    36 +        print(f"Missing log file: {path}")
    37 +        return
    38 +
    39 +    lat, play_lat, gcl, drift = [], [], [], []
    40 +    with path.open() as f:
    41 +        for line in f:
    42 +            try:
    43 +                rec = json.loads(line)
    44 +                if "latency_ms" in rec:
    45 +                    lat.append(float(rec["latency_ms"]))
    46 +                if "playback_latency_ms" in rec:
    47 +                    play_lat.append(float(rec["playback_latency_ms"]))
    48 +                if "gcl" in rec:
    49 +                    gcl.append(float(rec["gcl"]))
    50 +                if "drift" in rec:
    51 +                    drift.append(float(rec["drift"]))
    52 +            except Exception:
    53 +                continue
    54 +
    55 +    print(f"Loaded {len(lat)} entries from {path}")
    56 +    def line(label, data):
    57 +        if not data:
    58 +            print(f"{label}: n=0")
    59 +            return
    60 +        print(f"{label}: n={len(data)} p50={percentile(data,50):.1f} p95
        ={percentile(data,95):.1f} max={max(data):.1f}")
    61 +
    62 +    line("End-to-end latency (ms)", lat)
    63 +    line("Playback latency (ms)", play_lat)
    64 +    if gcl:
    65 +        print(f"GCL: mean={statistics.mean(gcl):.3f} min={min(gcl):.3f}"
        )
    66 +    if drift:
    67 +        print(f"Drift: p95={percentile(drift,95):.3f} max={max(drift):.3
        f}")
    68 +
    69 +    if HAVE_PLOT and lat:
    70 +        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    71 +        axes[0].hist(lat, bins=50, color="#1f77b4")
    72 +        axes[0].set_title("Latency (ms)")
    73 +        axes[1].hist(gcl, bins=30, color="#22c55e") if gcl else axes[1].
        text(0.5,0.5,"No GCL", ha="center")
    74 +        axes[1].set_title("GCL")
    75 +        axes[2].hist(drift, bins=30, color="#fb923c") if drift else axes
        [2].text(0.5,0.5,"No drift", ha="center")
    76 +        axes[2].set_title("Drift")
    77 +        plt.tight_layout()
    78 +        plt.show()
    79 +
    80 +if __name__ == "__main__":
    81 +    p = Path(os.environ.get("GOECKOH_MIRROR_METRICS_LOG", DEFAULT_PATH))
    82 +    main(p)

