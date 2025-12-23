def __init__(self, dim: int = 512):
    self.dim = dim
    self.graph = nx.hypercube_graph(dim)

def project(self, point: np.ndarray) -> np.ndarray:
    pca = PCA(n_components=3)
    return pca.fit_transform(point.reshape(1, -1))[0]

