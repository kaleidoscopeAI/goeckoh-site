def __init__(self,
             nodes: Sequence[int],
             edges: Sequence[Tuple[int, int]],
             Sigma_inv: np.ndarray,
             X_bar: np.ndarray,
             lambda_bit: float = 1.0,
             lambda_pos: float = 1.0,
             lambda_label: float = 1.0) -> None:
    self.nodes = list(nodes)
    self.edges = list(edges)
    self.Sigma_inv = np.asarray(Sigma_inv)
    self.X_bar = np.asarray(X_bar).ravel()
    self.lambda_bit = float(lambda_bit)
    self.lambda_pos = float(lambda_pos)
    self.lambda_label = float(lambda_label)
    self.neighbors: Dict[int, List[int]] = {n: [] for n in self.nodes}
    for (i, j) in self.edges:
        self.neighbors.setdefault(i, []).append(j)
        self.neighbors.setdefault(j, []).append(i)

def energy(self, state: HybridState) -> float:
    xs = [state.x[n].ravel() for n in sorted(state.x.keys())]
    if len(xs) == 0:
        cont = 0.0
    else:
        X_vec = np.concatenate(xs)
        if X_vec.size != self.X_bar.size or self.Sigma_inv.shape[0] != self.Sigma_inv.shape[1] or self.Sigma_inv.shape[0] != X_vec.size:
            raise ValueError("Sigma_inv or X_bar shape mismatch with state x")
        diff = X_vec - self.X_bar
        cont = 0.5 * float(diff.T @ self.Sigma_inv @ diff)
    pair_energy = 0.0
    for (i, j) in self.edges:
        Ei = np.asarray(state.E[i]).astype(int)
        Ej = np.asarray(state.E[j]).astype(int)
        ham = int(np.bitwise_xor(Ei, Ej).sum())
        pos_diff = np.asarray(state.x[i]) - np.asarray(state.x[j])
        pair_energy += self.lambda_bit * float(ham) + self.lambda_pos * float(np.sum(pos_diff ** 2))
    label_term = 0.0
    return float(cont + pair_energy + label_term)

def delta_energy_for_bitflip(self, state: HybridState, node: int, bit_idx: int) -> float:
    Ei = np.asarray(state.E[node]).astype(int)
    orig_bit = int(Ei[bit_idx])
    flipped = 1 - orig_bit
    delta = 0.0
    for j in self.neighbors.get(node, []):
        Ej = np.asarray(state.E[j]).astype(int)
        # change in Hamming distance at that bit
        before = 1 if orig_bit != int(Ej[bit_idx]) else 0
        after = 1 if flipped != int(Ej[bit_idx]) else 0
        delta += self.lambda_bit * float(after - before)
    return float(delta)

def analytic_gradient(self, state: HybridState) -> Dict[int, np.ndarray]:
    grads: Dict[int, np.ndarray] = {}
    xs = [state.x[n].ravel() for n in sorted(state.x.keys())]
    if len(xs) == 0:
        return {n: np.zeros_like(state.x[n]) for n in state.x}
    X_vec = np.concatenate(xs)
    grad_vec = self.Sigma_inv @ (X_vec - self.X_bar)
    sizes = [state.x[n].ravel().size for n in sorted(state.x.keys())]
    idx = 0
    for n, sz in zip(sorted(state.x.keys()), sizes):
        g = grad_vec[idx:idx + sz]
        grads[n] = np.asarray(g).reshape(state.x[n].shape)
        idx += sz
    for i in self.nodes:
        if i not in grads:
            grads[i] = np.zeros_like(state.x[i])
    for (i, j) in self.edges:
        diff = np.asarray(state.x[i]) - np.asarray(state.x[j])
        grads[i] = grads[i] + 2.0 * self.lambda_pos * diff
        grads[j] = grads[j] - 2.0 * self.lambda_pos * diff
    return grads

