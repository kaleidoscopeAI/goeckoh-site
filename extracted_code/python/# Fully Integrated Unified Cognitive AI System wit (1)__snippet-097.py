def __init__(self, n_nodes=64):  # Reduced for CPU/mobile
    self.nodes = [OrganicNode(i) for i in range(n_nodes)]
    self.graph = nx.Graph()
    for i in range(n_nodes):
        self.graph.add_node(i, node=self.nodes[i])
    for i in range(n_nodes):
        for j in range(i+1, n_nodes):
            if random.random() < 0.05:
                self.graph.add_edge(i, j)
                self.nodes[i].connect(self.nodes[j])
    self.env = CognitiveEnvironment()
    self.transformer = ReflectionTransformer(input_dim=8)  # Reduced dim

def iterate(self):
    self.env.fluctuate()
    for node in self.nodes:
        node.update(self.env)
        node.normalize()
        if random.random() < 0.01 and node.connections:
            other = random.choice(node.connections)
            node.bit.entangle(other.bit)
            node.arousal = node.bit.real
            other.arousal = other.bit.real

def cluster_supernodes(self, n_clusters=8):  # Reduced
    data = np.array([n.vector for n in self.nodes])
    kmeans = KMeans(n_clusters=n_clusters, n_init=10).fit(data)
    clusters = [[] for _ in range(n_clusters)]
    for idx, label in enumerate(kmeans.labels_):
        clusters[label].append(self.nodes[idx])
    return [SuperNode(cluster) for cluster in clusters]

