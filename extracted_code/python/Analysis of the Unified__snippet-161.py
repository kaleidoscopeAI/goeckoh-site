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

