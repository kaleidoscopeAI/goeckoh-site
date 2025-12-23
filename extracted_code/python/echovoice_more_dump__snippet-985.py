class CognitiveCube:
    def __init__(self, n_nodes=216):
        self.nodes = [ThoughtNode(np.random.randn(16)) for _ in range(n_nodes)]
        self.graph = nx.Graph()
        for i, node in enumerate(self.nodes):
            self.graph.add_node(i, node=node)
        for i in range(n_nodes):
            for j in range(i+1, n_nodes):
                if random.random() < 0.05:
                    self.graph.add_edge(i, j)
        self.env = CognitiveEnvironment()

    def iterate(self):
        self.env.fluctuate()
        for node in self.nodes:
            node.update(self.env)
            node.normalize()

    def cluster_supernodes(self, n_clusters=12):
        data = np.array([n.vector for n in self.nodes])
        kmeans = KMeans(n_clusters=n_clusters, n_init=10).fit(data)
        clusters = [[] for _ in range(n_clusters)]
        for idx, label in enumerate(kmeans.labels_):
            clusters[label].append(self.nodes[idx])
        return [SuperNode(cluster) for cluster in clusters]

