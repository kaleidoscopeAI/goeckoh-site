def __init__(self,
             nodes: Sequence[int],
             edges: Sequence[Tuple[int, int]],
             Sigma_inv: np.ndarray,
             X_bar: np.ndarray,
             lambda_bit: float = 1.0,
             lambda_pos: float = 1.0,
             lambda_label: float = 1.0) -> None:
    # ... (initialization)
    pass

def energy(self, state: HybridState) -> float:
    # 1. Continuous Energy: How far the system's vectors are from an ideal state.
    xs = [state.x[n].ravel() for n in sorted(state.x.keys())]
    X_vec = np.concatenate(xs)
    diff = X_vec - self.X_bar
    cont = 0.5 * float(diff.T @ self.Sigma_inv @ diff)

    # 2. Discrete & Relational Energy: The "stress" or "disagreement" between connected nodes.
    pair_energy = 0.0
    for (i, j) in self.edges:
        # Measures Hamming distance (XOR popcount) between node bit-strings.
        Ei = np.asarray(state.E[i]).astype(int)
        Ej = np.asarray(state.E[j]).astype(int)
        ham = int(np.bitwise_xor(Ei, Ej).sum())

        # Measures squared distance between node vector positions.
        pos_diff = np.asarray(state.x[i]) - np.asarray(state.x[j])

        pair_energy += self.lambda_bit * float(ham) + self.lambda_pos * float(np.sum(pos_diff ** 2))

    # Total energy is the sum of all components.
    return float(cont + pair_energy)

