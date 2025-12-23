"""
Self-replicating, self-correcting dynamical system on a graph lattice.

- Nodes carry scalar energy.
- Edges define a sparse connectivity graph.
- Energy evolves via logistic-like ODE with external input.
- High-energy nodes replicate.
- Self-reflection adjusts growth parameter and connectivity.
"""

def __init__(self, cfg: CrystalConfig):
    self.cfg = cfg
    self.graph = self._initialize_graph(cfg.num_nodes)
    self.energies = torch.tensor(
        [random.uniform(0.0, 10.0) for _ in range(cfg.num_nodes)],
        dtype=torch.float32,
    )
    self.params: Dict[str, float] = {"a": 1.0, "b": 0.1, "c": 0.5}
    self.history: List[float] = []

@self_correcting()
def _initialize_graph(self, num_nodes: int) -> nx.Graph:
    G = nx.Graph()
    for i in range(num_nodes):
        G.add_node(i)
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if random.random() < 0.03:  # sparser for large N
                G.add_edge(i, j)
    if not nx.is_connected(G):
        components = [list(c) for c in nx.connected_components(G)]
        base = components[0]
        for comp in components[1:]:
            u = random.choice(base)
            v = random.choice(comp)
            G.add_edge(u, v)
    return G

@self_correcting()
def update_from_rms(self, rms: float) -> None:
    """Update energies using a uniform external input derived from RMS."""
    external = float(rms)
    inputs = torch.full_like(self.energies, external)
    a = self.params["a"]
    b = self.params["b"]
    c = self.params["c"]
    dE = a * self.energies - b * (self.energies**2) + c * inputs
    self.energies = (self.energies + self.cfg.dt * dE).clamp(min=0.0)
    self.history.append(self.energies.mean().item())

@self_correcting()
def replicate_nodes(self) -> None:
    new_nodes: List[int] = []
    current_nodes = list(self.graph.nodes)
    for node in current_nodes:
        if self.energies[node] > self.cfg.energy_threshold:
            new_node = self.graph.number_of_nodes()
            self.graph.add_node(new_node)
            self.graph.add_edge(node, new_node)
            child_energy = self.energies[node] * self.cfg.replication_rate
            self.energies = torch.cat([self.energies, child_energy.view(1)])
            self.energies[node] *= 1.0 - self.cfg.replication_rate
            new_nodes.append(new_node)
    if new_nodes:
        print(f"[CRYSTAL] Replicated nodes: {new_nodes}")

@self_correcting()
def self_reflect(self) -> None:
    if len(self.history) <= 1:
        return
    growth = self.history[-1] - self.history[0]
    if growth < 0.0:
        print("[CRYSTAL] Mean energy decaying → increasing a slightly.")
        self.params["a"] += 0.1
    clustering = nx.average_clustering(self.graph)
    if clustering < 0.05:
        print("[CRYSTAL] Low clustering → adding edges.")
        nodes = list(self.graph.nodes)
        for _ in range(max(1, len(nodes) // 100)):
            i, j = random.sample(nodes, 2)
            if not self.graph.has_edge(i, j):
                self.graph.add_edge(i, j)

def get_gcl(self) -> float:
    return float(self.energies.mean().item())


