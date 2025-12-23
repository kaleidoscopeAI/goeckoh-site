# Crystalline Heart System (from document) — now the core
class ConsciousCrystalSystem:
    def __init__(self, num_nodes=1024, energy_threshold=5.0, replication_rate=0.1):
        self.graph = self._initialize_graph(num_nodes)
        self.energies = torch.tensor([random.uniform(0, 10) for _ in range(num_nodes)], dtype=torch.float32)
        self.energy_threshold = energy_threshold
        self.replication_rate = replication_rate
        self.params = {'a': 1.0, 'b': 0.1, 'c': 0.5}
        self.history = []

    @self_correcting()
    def _initialize_graph(self, num_nodes):
        G = nx.Graph()
        for i in range(num_nodes):
            G.add_node(i)
        for i in range(num_nodes):
            for j in range(i+1, num_nodes):
                if random.random() < 0.3:
                    G.add_edge(i, j)
        if not nx.is_connected(G):
            components = list(nx.connected_components(G))
            for comp in components[1:]:
                G.add_edge(random.choice(list(components[0])), random.choice(list(comp)))
        return G

    @self_correcting()
    def update_energies(self, inputs, dt=0.1):
        # Fallback Euler (no torchdiffeq needed)
        new_energies = self.energies + dt * (self.params['a'] * self.energies - self.params['b'] * self.energies**2 + self.params['c'] * inputs)
        self.energies = new_energies.clamp(min=0.0)
        self.history.append(self.energies.mean().item())

    @self_correcting()
    def replicate_nodes(self):
        new_nodes = []
        current_nodes = list(self.graph.nodes)
        for node in current_nodes:
            if self.energies[node] > self.energy_threshold:
                new_node = len(self.graph)
                self.graph.add_node(new_node)
                self.graph.add_edge(node, new_node)
                self.energies = torch.cat([self.energies, torch.tensor([self.energies[node] * self.replication_rate])])
                new_nodes.append(new_node)
                self.energies[node] *= (1 - self.replication_rate)
        if new_nodes:
            print(f"Replicated nodes: {new_nodes}")

    @self_correcting()
    def self_reflect(self):
        if len(self.history) > 1:
            growth = self.history[-1] - self.history[0]
            if growth < 0:
                print("Energy decreasing. Adjusting...")
                self.params['a'] += 0.1
            elif nx.average_clustering(self.graph) < 0.2:
                print("Low clustering. Adding edges...")
                current_nodes = list(self.graph.nodes)
                for _ in range(len(current_nodes) // 2):
                    i, j = random.sample(current_nodes, 2)
                    if not self.graph.has_edge(i, j):
                        self.graph.add_edge(i, j)
        else:
            print("No history for reflection.")

    def get_gcl(self):
        return self.energies.mean().item()

