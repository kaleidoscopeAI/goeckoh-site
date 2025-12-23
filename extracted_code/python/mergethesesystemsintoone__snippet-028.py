class Phase:
    RAW="raw"; GEL="gel"; CRYSTAL="crystal"

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

def anneal_schedule(sigma0: float, gamma: float, step: int, sigma_min: float) -> float:
    return max(sigma0 * (gamma ** step), sigma_min)

def expected_cos_with_noise(ei: np.ndarray, ej: np.ndarray, sigma: float, M: int = 4) -> float:
    rng = np.random.RandomState(11)
    sims = []
    for _ in range(M):
        ei_n = ei + sigma * rng.normal(0.0, 1.0, size=ei.shape); ei_n /= (np.linalg.norm(ei_n)+1e-9)
        ej_n = ej + sigma * rng.normal(0.0, 1.0, size=ej.shape); ej_n /= (np.linalg.norm(ej_n)+1e-9)
        sims.append(float(np.dot(ei_n, ej_n)))
    return float(np.mean(sims))

def weights_tension(E: np.ndarray, edges: np.ndarray, sigma: float, M: int = 3) -> Tuple[np.ndarray, np.ndarray]:
    w = np.zeros(len(edges), dtype=np.float64)
    for k, (i, j) in enumerate(edges):
        w[k] = expected_cos_with_noise(E[i], E[j], sigma=sigma, M=M)
    tau = 1.0 - w
    return w, tau

def energetics(E: np.ndarray, S: np.ndarray, edges: np.ndarray, sigma: float) -> dict:
    N = E.shape[0]
    if len(edges) == 0:
        return {"H_bits": float(np.mean(1.0 - S) if N else 0.0), "S_field": 0.0, "L": 0.0}
    w, tau = weights_tension(E, edges, sigma=sigma)
    H_bits = float(np.mean(1.0 - S) if N else 0.0)
    S_field = float(np.mean(tau))
    L = float(np.sum(tau * tau))
    return {"H_bits": H_bits, "S_field": S_field, "L": L}

