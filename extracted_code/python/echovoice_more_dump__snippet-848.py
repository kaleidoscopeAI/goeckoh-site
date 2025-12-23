def step_update(nodes: List[Node], cfg: Config, step: int) -> None:
    n = len(nodes)
    B = compute_B_matrix(nodes, cfg)
    W = compute_W_matrix(nodes, cfg)

    # compute pairwise differences quickly
    Nvecs = np.stack([np.concatenate((node.K, node.Psi, node.D)) for node in nodes])
    # For simplicity, treat N as concatenation of K,Psi,D in dynamics

    # compute gradients / data changes
    delta_D = [mock_crawl_source(node, cfg, step) for node in nodes]

    # compute sigma, Phi, Spec for each node
    for i, node in enumerate(nodes):
        # sigma: sum_j B_ij (N_j - N_i) + kappa U + theta grad D
        Nj_minus_Ni = np.zeros_like(node.K)
        for j in range(n):
            if i == j:
                continue
            Nj_minus_Ni += B[i, j] * (nodes[j].K - node.K)
        node.sigma = Nj_minus_Ni + cfg.kappa * node.U[: node.K.shape[0]] + cfg.theta * delta_D[i]
        # Phi: N - N_hat - gamma_U U
        # Here N_hat is a simple prediction: moving average of neighbors' K
        neighbor_mean = np.mean([nodes[j].K for j in range(n) if j != i], axis=0)
        node.Phi = node.K - neighbor_mean - cfg.lambda_U * node.U[: node.K.shape[0]]
        node.Spec = speculation_operator(node, cfg)

    # update U, M, D, and N (K,Psi,D) according to equations
    for i, node in enumerate(nodes):
        # U update
        psi_sum = np.zeros_like(node.U)
        for j in range(n):
            psi_sum += B[i, j] * nodes[j].U
        node.U = node.U + cfg.lambda_U * (decoherence(node) * np.ones_like(node.U) + psi_sum + cfg.zeta * np.linalg.norm(delta_D[i]))
        # M update
        mimicry_sum = np.zeros_like(node.M)
        for j in range(n):
            mimicry_sum += W[i, j] * (nodes[j].K - node.K)
        node.M = node.M + cfg.lambda_M * mimicry_sum
        # D update (data ingestion) - encode delta into D space
        node.D = node.D + cfg.lambda_D * delta_D[i]

    # Finally update K (as proxy for N) using integrated update
    for i, node in enumerate(nodes):
        stress_term = np.zeros_like(node.K)
        for j in range(n):
            stress_term += B[i, j] * (nodes[j].K - node.K)
        mimicry_term = cfg.lambda_M * sum(W[i, j] * (nodes[j].K - node.K) for j in range(n))
        node.K = (
            node.K
            + cfg.eta * stress_term
            + cfg.lambda_phi * node.Phi
            + cfg.gamma * node.Spec
            - cfg.delta * node.U[: node.K.shape[0]]
            + mimicry_term
        )

    # small normalization / noise to keep stable
    for node in nodes:
        node.K = node.K / (np.linalg.norm(node.K) + 1e-12)
        node.D = node.D / (np.linalg.norm(node.D) + 1e-12 + 1e-12 * random.random())


