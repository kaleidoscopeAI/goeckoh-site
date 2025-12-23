def decoherence(node: Node) -> float:
    # quadratic on misalignment magnitude
    if node.Phi is None:
        return 0.0
    return np.linalg.norm(node.Phi) ** 2


def speculation_operator(node: Node, cfg: Config) -> np.ndarray:
    # simple speculative vector: small nonlinear transform of K + D
    z = np.tanh(node.K + node.D - node.U * 0.01)
    return z * cfg.gamma


