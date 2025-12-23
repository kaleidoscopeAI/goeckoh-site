def role_rho(a: str, b: str) -> float:
    # cooperative if same role and blue/blue bias, adversarial if red vs blue
    if a == b:
        return 1.0
    if (a == 'red' and b == 'blue') or (a == 'blue' and b == 'red'):
        return -1.0
    return 0.0


def compute_B_matrix(nodes: List[Node], cfg: Config) -> np.ndarray:
    n = len(nodes)
    B = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            KiKj = np.dot(nodes[i].K, nodes[j].K)
            DiDj = np.dot(nodes[i].D, nodes[j].D)
            denom = (np.linalg.norm(nodes[i].K) * np.linalg.norm(nodes[j].K)
                     + np.linalg.norm(nodes[i].D) * np.linalg.norm(nodes[j].D) + 1e-12)
            g = (KiKj + DiDj) / denom
            psi_diff = np.linalg.norm(nodes[i].Psi - nodes[j].Psi)
            g *= math.exp(-cfg.alpha * (psi_diff ** 2))
            rho = role_rho(nodes[i].R, nodes[j].R)
            phi = np.linalg.norm(nodes[i].U - nodes[j].U) / (np.linalg.norm(nodes[i].U) + np.linalg.norm(nodes[j].U) + 1e-12)
            B[i, j] = g * (1 - phi) * rho
    return B


def compute_W_matrix(nodes: List[Node], cfg: Config) -> np.ndarray:
    n = len(nodes)
    W = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            ck = cos_sim(nodes[i].K, nodes[j].K)
            cp = cos_sim(nodes[i].Psi, nodes[j].Psi)
            cd = cos_sim(nodes[i].D, nodes[j].D)
            w = (ck + cp + cd) / 3.0
            w *= math.exp(-cfg.beta * np.linalg.norm(nodes[j].U))
            # role bias: prefer learning from same-role or trusted
            role_bias = 1.0 if nodes[i].R == nodes[j].R else 0.5
            W[i, j] = w * role_bias
    return W


