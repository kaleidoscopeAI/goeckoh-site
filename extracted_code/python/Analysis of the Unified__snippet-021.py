# hybrid_viz.py
"""
Simple visualization for hybrid run history.
Accepts job record from hybrid_orchestrator._JOBS[job_id] or the demo 'history'
Produces:
 - time series plot: mix, purity, entropy
 - stacked probability plot
 - heatmaps of |R|^2 and |Q|^2 for final snapshot
"""

import matplotlib.pyplot as plt
import numpy as np
import json
from typing import List, Dict

def plot_history(history: List[Dict], save_prefix: str = "hybrid"):
    times = [h["t"] for h in history]
    mixes = [h["diag"].get("mix", 0.0) for h in history]
    purities = [h["diag"].get("purity", 0.0) for h in history]
    entropies = [h["diag"].get("entropy", 0.0) for h in history]
    probs = np.vstack([h["p"] for h in history])

    fig, axs = plt.subplots(3,1, figsize=(8,10), sharex=True)
    axs[0].plot(times, mixes, label="mix (sum→product)"); axs[0].legend(); axs[0].set_ylabel("mix")
    axs[1].plot(times, purities, label="purity"); axs[1].plot(times, entropies, label="entropy"); axs[1].legend(); axs[1].set_ylabel("value")
    axs[2].stackplot(times, probs.T, labels=[f"p{i}" for i in range(probs.shape[1])]); axs[2].legend(); axs[2].set_ylabel("prob")
    axs[2].set_xlabel("time")
    plt.tight_layout()
    plt.savefig(f"{save_prefix}_timeseries.png", dpi=200)
    plt.close(fig)
    print("Wrote", f"{save_prefix}_timeseries.png")

def heatmap_from_final(core, save_prefix="hybrid"):
    R = core.R
    Q = core.Q
    I_R = np.abs(R)**2
    I_Q = np.abs(Q)**2
    fig, axs = plt.subplots(1,2, figsize=(8,4))
    im0 = axs[0].imshow(I_R, origin='lower', interpolation='nearest')
    axs[0].set_title("|R|^2")
    plt.colorbar(im0, ax=axs[0], fraction=0.046)
    im1 = axs[1].imshow(I_Q, origin='lower', interpolation='nearest')
    axs[1].set_title("|Q|^2")
    plt.colorbar(im1, ax=axs[1], fraction=0.046)
    plt.tight_layout()
    plt.savefig(f"{save_prefix}_heatmaps.png", dpi=200)
    plt.close()
    print("Wrote", f"{save_prefix}_heatmaps.png")

# Quick helper to load job history (structure produced by hybrid_orchestrator)
def load_history_from_job(job_record: Dict):
    return job_record.get("history", [])

# If run as script, expect a JSON history file
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python hybrid_viz.py <history_json>")
        sys.exit(1)
    fname = sys.argv[1]
    with open(fname, "r") as f:
        history = json.load(f)
    # history entries expected: {t, diag, p}
    plot_history(history, save_prefix="hybrid_run")
    print("Done")
