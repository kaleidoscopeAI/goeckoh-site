"""
Run a small 2x2 demo demonstrating hybrid mixing and stress dynamics.
Saves diagnostics to a list and returns final core.
"""
# initial R (two system outcomes, two apparatus columns)
R0 = np.array([[0.8+0.0j, 0.1+0.0j],
               [0.2+0.0j, 0.9+0.0j]], dtype=np.complex128)
core = HybridRelationalCore(R0, mix=0.0, gamma=0.5, product_method="softlogprod", product_params={"alpha":1.0,"beta":3.0})
# simple Hamiltonians (Pauli-like)
Hs = np.array([[0.0, 1.0],[1.0, 0.0]], dtype=np.complex128)
Ha = np.array([[0.5, 0.0],[0.0, -0.5]], dtype=np.complex128)
Hint = None

history = []
# simple controller: gradually increase mix if max_stress > thresh; otherwise reduce mix slowly
def simple_controller(diag):
    adv = 0.0
    if diag["max_stress"] > 0.1:
        adv = 0.02
    else:
        adv = -0.005
    # adjust gamma based on purity to stabilize
    gamma_delta = 0.0
    if diag["purity"] < 0.95:
        gamma_delta = 0.01
    return {"mix_delta": adv, "gamma_delta": gamma_delta, "bond_scale": 1.0, "product_params": {"alpha":1.0, "beta":3.0}}

for t in range(steps):
    # optionally perturb R with small noise to simulate streaming data
    if noisy and (t % 15 == 0):
        noise = (np.random.randn(*core.R.shape) + 1j * np.random.randn(*core.R.shape)) * 0.005
        core.R += noise
    # evolve
    core.evolve_step(Hs, Ha, Hint, dt)
    # adapt bonds
    core.adapt_bonds_hebb(eta=1e-4, decay=1e-5)
    # ai step every few iterations
    if t % 5 == 0:
        core.ai_control_step(simple_controller)
    if t % log_every == 0 or t == steps-1:
        diag = core.diagnostics()
        p = core.measure_probs()
        history.append({"t": t*dt, "diag": diag, "p": p.copy(), "R": core.R.copy(), "Q": core.Q.copy(), "B": core.B.copy()})
        logger.info("t=%.3f mix=%.3f purity=%.4f entropy=%.4f probs=%s", t*dt, core.mix, diag["purity"], diag["entropy"], np.array2string(p, precision=3))
return core, history

