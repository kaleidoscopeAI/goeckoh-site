def expected_cos_with_noise(ei: np.ndarray, ej: np.ndarray, sigma: float, M: int = 4) -> float:
    rng = np.random.RandomState(11)
    sims = []
    for _ in range(M):
        ei_n = ei + sigma * rng.normal(0.0, 1.0, size=ei.shape)
        ej_n = ej + sigma * rng.normal(0.0, 1.0, size=ej.shape)
        ei_n = ei_n / (np.linalg.norm(ei_n) + 1e-9)
        ej_n = ej_n / (np.linalg.norm(ej_n) + 1e-9)
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

