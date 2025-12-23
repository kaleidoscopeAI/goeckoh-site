# sandbox_tester.py
"""
Sandbox evaluation: run short rollouts on a copy of a HybridRelationalCore,
compute average metrics: avg_purity, avg_entropy, avg_max_stress, ok(bool).
Designed for fast checks inside apply_action_safe.
"""

from __future__ import annotations
import numpy as np
import copy
from typing import Dict, Any

def sandbox_evaluate(core, Hs, Ha, Hint=None, steps: int = 8, dt: float = 0.01) -> Dict[str, Any]:
    c = copy.deepcopy(core)
    purities = []
    entropies = []
    max_stresses = []
    ok = True
    for _ in range(steps):
        try:
            c.evolve_step(Hs, Ha, Hint, dt)
            c.adapt_bonds_hebb(eta=1e-5, decay=1e-6)
            d = c.diagnostics()
            purities.append(d["purity"])
            entropies.append(d["entropy"])
            max_stresses.append(d["max_stress"])
            # sanity checks
            if not np.isfinite(d["purity"]) or not np.isfinite(d["entropy"]):
                ok = False
                break
            # detect catastrophes: purity extremely low or enormous bond
            if d["purity"] < 1e-6 or np.max(c.B) > 1e8:
                ok = False
                break
        except Exception:
            ok = False
            break
    metrics = {
        "avg_purity": float(np.mean(purities) if purities else 0.0),
        "avg_entropy": float(np.mean(entropies) if entropies else 0.0),
        "avg_max_stress": float(np.mean(max_stresses) if max_stresses else 0.0),
        "ok": bool(ok)
    }
    return metrics
