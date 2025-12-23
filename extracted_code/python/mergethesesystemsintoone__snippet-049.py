class Phase:
    RAW = "raw"
    GEL = "gel"
    CRYSTAL = "crystal"

def cos_sim(a: np.ndarray, b: np.ndarray, eps: float = 1e-9) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + eps))

def knn_indices(E: np.ndarray, i: int, k: int = 8) -> List[int]:
    x = E[i]
    sims = (E @ x) / (np.linalg.norm(E, axis=1) * (np.linalg.norm(x) + 1e-9) + 1e-12)
    order = np.argsort(-sims)
    return [j for j in order if j != i][:k]

def monte_carlo_variance(E: np.ndarray, i: int, k: int, sigma: float, M: int = 6, rng=None) -> float:
    if rng is None:
        rng = np.random.RandomState(7)
    idx = knn_indices(E, i, k=max(1, min(k, E.shape[0]-1)))
    vals = []
    D = E.shape[1]
    for _ in range(M):
        ei = E[i] + sigma * rng.normal(0.0, 1.0, size=D)
        ei = ei / (np.linalg.norm(ei) + 1e-9)
        sims = []
        for j in idx:
            ej = E[j] + sigma * rng.normal(0.0, 1.0, size=D)
            ej = ej / (np.linalg.norm(ej) + 1e-9)
            sims.append(cos_sim(ei, ej))
        vals.append(max(sims) if sims else 0.0)
    return float(np.var(vals))

def stability_score(var_sigma: float) -> float:
    return 1.0 / (1.0 + var_sigma)

def phase_transition(var_sigma: float, sigma: float, theta_gel: float, theta_crystal: float, sigma_min: float) -> str:
    if var_sigma < theta_crystal and sigma <= sigma_min:
        return Phase.CRYSTAL
    if var_sigma < theta_gel:
        return Phase.GEL
    return Phase.RAW

def anneal_schedule(sigma0: float, gamma: float, step: int, sigma_min: float) -> float:
    return max(sigma0 * (gamma ** step), sigma_min)

