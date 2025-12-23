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

