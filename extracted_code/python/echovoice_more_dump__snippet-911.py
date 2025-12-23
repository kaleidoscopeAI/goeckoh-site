def create_nodes(count: int, cfg: Config, seed: int = 0) -> List[Node]:
    rng = np.random.RandomState(seed)
    nodes = []
    roles = ['red', 'blue', 'crawler', 'analyzer']
    for i in range(count):
        K = rng.normal(size=cfg.dims)
        K = K / (np.linalg.norm(K) + 1e-12)
        Psi = rng.normal(size=cfg.dims)
        Psi = Psi / (np.linalg.norm(Psi) + 1e-12)
        D = rng.normal(size=cfg.dims)
        D = D / (np.linalg.norm(D) + 1e-12)
        U = rng.normal(scale=0.01, size=cfg.dims)
        M = np.zeros(cfg.dims)
        X = rng.normal(size=cfg.dims)
        node = Node(
            id=i,
            X=X,
            S=1.0,
            E=1.0,
            K=K,
            Psi=Psi,
            U=U,
            M=M,
            R=roles[i % len(roles)],
            D=D,
        )
        nodes.append(node)
    return nodes


