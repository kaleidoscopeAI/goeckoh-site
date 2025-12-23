# hybrid_orchestrator.py
"""
Hybrid Orchestrator: integrates HybridRelationalCore into ARQIS orchestration,
provides safe sandbox testing + controlled commit, exposes simple API:
 - start_hybrid_job(sanitized_job, controller_source='manual'|'llm')
 - apply_action_safe(job_id, action_dict, run_sandbox=True)
This module depends on:
 - relational_hybrid_engine.HybridRelationalCore
 - arqis.orchestrator (for persistence/orchestration primitives) or standalone usage
 - sandbox_tester.sandbox_evaluate
 - llm_bridge.LLMBridge (optional)
"""

from __future__ import annotations
import logging
import time
import copy
import uuid
from typing import Dict, Any, Optional

import numpy as np

from relational_hybrid_engine import HybridRelationalCore  # from file earlier
try:
    from arqis.orchestrator import start_job as arqis_start_job
    ARQIS_PRESENT = True
except Exception:
    ARQIS_PRESENT = False

from sandbox_tester import sandbox_evaluate
from llm_bridge import LLMBridge

logger = logging.getLogger("hybrid_orch")
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s", "%H:%M:%S"))
logger.addHandler(handler)
logger.setLevel(logging.INFO)

# ---------------------------
# Job registry (in-memory small)
# ---------------------------
_JOBS: Dict[str, Dict[str, Any]] = {}

def _gen_run_id():
    return f"hybrid_{uuid.uuid4().hex[:10]}"

# ---------------------------
# Start a hybrid job (simple)
# ---------------------------
def start_hybrid_job_from_target(target_obj: Any, policy: Dict[str, Any] = None) -> str:
    """
    Create a HybridRelationalCore from a target object (like a classical vector or matrix)
    and store job metadata. Returns job_id.
    Minimally builds default Hs/Ha similar to injector.build_default_hamiltonians.
    """
    policy = policy or {}
    # Determine shape & embed simply: if 1D -> classical, if 2D -> assume R
    if isinstance(target_obj, (list, tuple)):
        arr = np.asarray(target_obj, dtype=float).flatten()
        R = np.outer(np.sqrt(arr/arr.sum()), np.array([1.0], dtype=np.complex128))
    else:
        arr = np.asarray(target_obj)
        if arr.ndim == 1:
            a = arr / np.sum(arr)
            R = np.outer(np.sqrt(a), np.array([1.0], dtype=np.complex128))
        elif arr.ndim == 2:
            R = np.asarray(arr, dtype=np.complex128)
        else:
            raise ValueError("Unsupported target shape")
    mix = float(policy.get("mix", 0.0))
    gamma = float(policy.get("gamma", 0.1))
    product_method = policy.get("product_method", "softlogprod")
    product_params = policy.get("product_params", {"alpha":1.0, "beta":3.0})
    core = HybridRelationalCore(R, mix=mix, gamma=gamma, product_method=product_method, product_params=product_params)

    # default Hs, Ha (small)
    dS, dA = core.R.shape
    if dS == 2:
        Hs = np.array([[0.0,1.0],[1.0,0.0]], dtype=np.complex128)
    else:
        Hs = np.eye(dS, dtype=np.complex128)
    if dA == 2:
        Ha = np.array([[0.5,0.0],[0.0,-0.5]], dtype=np.complex128)
    else:
        Ha = np.eye(dA, dtype=np.complex128) * 0.1

    job_id = _gen_run_id()
    _JOBS[job_id] = {"job_id": job_id, "core": core, "Hs": Hs, "Ha": Ha, "Hint": None, "policy": policy, "history": []}
    logger.info("Hybrid job %s created dims=%s", job_id, core.R.shape)
    return job_id

# ---------------------------
# Get job / step / snapshot functions
# ---------------------------
def step_job(job_id: str, dt: float = 0.01, steps: int = 1):
    rec = _JOBS[job_id]
    core: HybridRelationalCore = rec["core"]
    Hs, Ha, Hint = rec["Hs"], rec["Ha"], rec["Hint"]
    for _ in range(steps):
        core.evolve_step(Hs, Ha, Hint, dt)
        core.adapt_bonds_hebb(eta=rec["policy"].get("bond_eta", 1e-4), decay=rec["policy"].get("bond_decay", 1e-5))
        # snapshot & persist minimally in memory
        rec["history"].append({"t": time.time(), "diag": core.diagnostics(), "p": core.measure_probs().tolist()})
    return rec["history"][-1]

def get_job_diag(job_id: str):
    rec = _JOBS[job_id]
    return rec["core"].diagnostics()

# ---------------------------
# Apply action safe: sandbox test and commit
# ---------------------------
def apply_action_safe(job_id: str, action: Dict[str, Any], sandbox_steps: int = 8, dt: float = 0.01,
                      min_reward_improve: float = 1e-4) -> Dict[str, Any]:
    """
    action: dict in allowed schema:
      {"mix_delta":..., "gamma_delta":..., "bond_scale":..., "product_params": {"alpha":..., "beta":...}}
    Returns dict with keys: approved(bool), reason, sandbox_metrics
    """
    rec = _JOBS[job_id]
    core: HybridRelationalCore = rec["core"]
    Hs, Ha, Hint = rec["Hs"], rec["Ha"], rec["Hint"]

    # copy core for sandbox
    sand_core = copy.deepcopy(core)
    # apply action to sandbox
    def apply_action_to_core(c: HybridRelationalCore, a: Dict[str, Any]):
        # mimic ai_control_step without logger
        mix_delta = float(a.get("mix_delta", 0.0))
        gamma_delta = float(a.get("gamma_delta", 0.0))
        bond_scale = float(a.get("bond_scale", 1.0))
        product_params = a.get("product_params", None)
        c.mix = float(np.clip(c.mix + mix_delta, 0.0, 1.0))
        c.gamma = float(np.clip(c.gamma + gamma_delta, 0.0, 10.0))
        if not np.isfinite(bond_scale) or bond_scale <= 0:
            bond_scale = 1.0
        c.B = np.clip(c.B * bond_scale, 0.0, 1e6)
        if isinstance(product_params, dict):
            a_ = float(product_params.get("alpha", c.product_params.get("alpha",1.0)))
            b_ = float(product_params.get("beta", c.product_params.get("beta",1.0)))
            c.product_params["alpha"] = np.clip(a_, 0.0, 5.0)
            c.product_params["beta"] = np.clip(b_, 0.0, 10.0)
    apply_action_to_core(sand_core, action)

    # run sandbox evaluation
    metrics = sandbox_evaluate(sand_core, Hs, Ha, Hint, steps=sandbox_steps, dt=dt)

    # baseline metrics: run baseline sandbox from original core for same rollout
    baseline_core = copy.deepcopy(core)
    baseline_metrics = sandbox_evaluate(baseline_core, Hs, Ha, Hint, steps=sandbox_steps, dt=dt)

    # compute reward delta (simple: improvement in purity - increase in entropy penalized)
    def score(m):
        # higher purity and lower entropy is considered good, also prefer reduction in max_stress
        return (m["avg_purity"] - 0.5*m["avg_entropy"]) - 0.1 * m["avg_max_stress"]
    score_base = score(baseline_metrics)
    score_new = score(metrics)
    delta = score_new - score_base

    approved = False
    reason = "rejected by policy"
    if delta >= min_reward_improve and metrics["ok"]:
        # commit: apply action to live core atomically
        apply_action_to_core(core, action)
        approved = True
        reason = f"approved (delta={delta:.6f})"
    else:
        reason = f"rejected (delta={delta:.6f})"

    result = {"approved": approved, "reason": reason, "delta": float(delta),
              "sandbox_metrics": metrics, "baseline_metrics": baseline_metrics}
    # log
    logger.info("Action %s on job %s -> %s", action, job_id, reason)
    # record action and result in job history
    rec.setdefault("actions", []).append({"action": action, "result": result, "t": time.time()})
    return result

# ---------------------------
# LLM integration helper (high level)
# ---------------------------
def propose_and_apply_from_llm(job_id: str, llm: LLMBridge, prompt_extra: str = "", sandbox_steps: int = 8, dt: float = 0.01):
    """
    Ask LLM for a suggestion for the job's diagnostics and try to apply it safely.
    The LLM output is parsed; if it's invalid JSON or fails validation nothing is applied.
    """
    rec = _JOBS[job_id]
    diag = rec["core"].diagnostics()
    prompt = f"Diagnostics:\n{diag}\n\nReturn a JSON action with keys: mix_delta, gamma_delta, bond_scale, product_params.\n{prompt_extra}"
    suggestion_text = llm.generate_prompt(prompt)
    action = llm.parse_action_json(suggestion_text)
    if action is None:
        return {"approved": False, "reason": "LLM produced no valid action", "raw": suggestion_text}
    # sanitize action to simple numeric fields
    safe_action = {}
    for k in ["mix_delta", "gamma_delta", "bond_scale"]:
        if k in action:
            try:
                safe_action[k] = float(action[k])
            except Exception:
                pass
    if "product_params" in action and isinstance(action["product_params"], dict):
        safe_action["product_params"] = {"alpha": float(action["product_params"].get("alpha",1.0)),
                                        "beta": float(action["product_params"].get("beta",1.0))}
    # apply safe
    return apply_action_safe(job_id, safe_action, sandbox_steps=sandbox_steps, dt=dt)

# ---------------------------
# Example tiny CLI (for quick tests)
# ---------------------------
if __name__ == "__main__":
    # quick demo: start job and let local controller propose
    job = start_hybrid_job_from_target([0.4, 0.6], policy={"mix":0.0, "gamma":0.4})
    print("Started job:", job)
    # step a bit
    print("Step:", step_job(job, dt=0.01, steps=20))
    # create an LLMBridge pointing to local ollama (optional)
    try:
        llm = LLMBridge(base_url="http://localhost:11434", model="mistral")
    except Exception:
        llm = None
    # if LLM available, call propose_and_apply_from_llm
    if llm:
        res = propose_and_apply_from_llm(job, llm)
        print("LLM propose result:", res)
    else:
        print("LLM not available; apply a manual small action")
        r = apply_action_safe(job, {"mix_delta": 0.05, "gamma_delta": 0.01}, sandbox_steps=8)
        print("Manual apply result:", r)
