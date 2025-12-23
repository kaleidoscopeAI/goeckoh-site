class Bond:
    a: int
    b: int
    k: float = 0.15
    rest: float = 1.0

class Cube:
    def __init__(self, n_per_edge: int = 6, seed: int = 42):
        np.random.seed(seed)
        self.G = nx.Graph()
        self.tick = 0
        idc = 0
        for x in range(n_per_edge):
            for y in range(n_per_edge):
                for z in range(n_per_edge):
                    p = np.array([x, y, z], dtype=float)
                    p = 2 * (p / (n_per_edge - 1)) - 1
                    self.G.add_node(idc, node=Node(id=idc, pos=p, fixed=False))
                    idc += 1
        def idx(x, y, z): return x * (n_per_edge**2) + y * n_per_edge + z
        for x in range(n_per_edge):
            for y in range(n_per_edge):
                for z in range(n_per_edge):
                    u = idx(x, y, z)
                    for dx, dy, dz in [(1, 0, 0), (0, 1, 0), (0, 0, 1)]:
                        nx_ = x + dx
                        ny_ = y + dy
                        nz_ = z + dz
                        if nx_ < n_per_edge and ny_ < n_per_edge and nz_ < n_per_edge:
                            v = idx(nx_, ny_, nz_)
                            self.G.add_edge(u, v, bond=Bond(a=u, b=v, k=0.15, rest=2 / (n_per_edge - 1)))
        corners = [0, idx(n_per_edge - 1, 0, 0), idx(0, n_per_edge - 1, 0), idx(0, 0, n_per_edge - 1),
                   idx(n_per_edge - 1, n_per_edge - 1, 0), idx(n_per_edge - 1, 0, n_per_edge - 1),
                   idx(0, n_per_edge - 1, n_per_edge - 1), idx(n_per_edge - 1, n_per_edge - 1, n_per_edge - 1)]
        for c in corners:
            self.G.nodes[c]['node'].fixed = True

    def step(self, dt: float = 0.1, damp: float = 0.9):
        forces = {i: np.zeros(3) for i in self.G.nodes}
        for u, v, data in self.G.edges(data=True):
            b: Bond = data['bond']
            pu = self.G.nodes[u]['node'].pos
            pv = self.G.nodes[v]['node'].pos
            d = pv - pu
            L = float(np.linalg.norm(d) + 1e-8)
            F = b.k * (L - b.rest) * (d / L)
            forces[u] += F
            forces[v] -= F
        for i, data in self.G.nodes(data=True):
            n: Node = data['node']
            if n.fixed:
                continue
            n.pos += dt * forces[i]
            n.pos *= damp
        self.tick += 1

    def metrics(self) -> Dict[str, float]:
        tension = 0.0
        energy = 0.0
        for u, v, data in self.G.edges(data=True):
            b: Bond = data['bond']
            pu = self.G.nodes[u]['node'].pos
            pv = self.G.nodes[v]['node'].pos
            L = float(np.linalg.norm(pv - pu))
            tension += abs(L - b.rest)
            energy += 0.5 * b.k * (L - b.rest)**2
        m = max(1, self.G.number_of_edges())
        return {"tension": tension / m, "energy": energy / m, "size": self.G.number_of_nodes()}

    def apply_adjustments(self, adj: Dict[str, float]):
        ks = float(adj.get("k_scale", 1.0))
        rs = float(adj.get("rest_scale", 1.0))
        ks = max(0.25, min(ks, 4.0))
        rs = max(0.5, min(rs, 1.5))
        for _, _, data in self.G.edges(data=True):
            b: Bond = data['bond']
            b.k *= ks
            b.rest *= rs

