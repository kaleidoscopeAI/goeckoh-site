def mock_encode_text_to_vec(text: str, dims: int) -> np.ndarray:
    # deterministic pseudo-encoder for reproducibility
    rnd = np.frombuffer(text.encode('utf8'), dtype=np.uint8)
    h = np.zeros(dims, dtype=float)
    for i, v in enumerate(rnd):
        h[i % dims] += (v + 1) * 0.001
    # normalize
    n = np.linalg.norm(h) + 1e-12
    return h / n


def mock_crawl_source(node: Node, cfg: Config, step: int) -> np.ndarray:
    # produce a synthetic 'crawl' delta X based on role and step
    seed = int(node.id * 1009 + step * 97)
    rng = np.random.RandomState(seed)
    delta = rng.normal(scale=0.1, size=cfg.dims)
    # small signal depending on role
    role_signal = {'red': -0.2, 'blue': 0.2, 'crawler': 0.5, 'analyzer': 0.1}
    delta += role_signal.get(node.R, 0.0)
    return delta


