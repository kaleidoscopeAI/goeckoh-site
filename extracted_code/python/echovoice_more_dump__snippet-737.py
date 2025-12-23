def build_minimal_system(node_count: int = 8, bit_dim: int = 64) -> Tuple[HybridState, SemanticHamiltonian, GradientFlow, MetropolisEngine]:
    rng = np.random.default_rng(42)
    E = {i: rng.integers(0, 2, size=(bit_dim,), dtype=np.uint8) for i in range(node_count)}
    x = {i: rng.standard_normal(3,) for i in range(node_count)}
    state = HybridState(E=E, x=x)
    state.pack_all()
    Sigma_inv = np.eye(3 * node_count)
    X_bar = np.zeros((3 * node_count,), dtype=float)
    edges = [(i, (i + 1) % node_count) for i in range(node_count)]
    ham = SemanticHamiltonian(nodes=list(range(node_count)), edges=edges, Sigma_inv=Sigma_inv, X_bar=X_bar,
                              lambda_bit=1.0, lambda_pos=0.1)
    gf = GradientFlow(state, ham, lr=0.05)
    me = MetropolisEngine(state, ham, hajek_schedule(c=1.0))
    return state, ham, gf, me

