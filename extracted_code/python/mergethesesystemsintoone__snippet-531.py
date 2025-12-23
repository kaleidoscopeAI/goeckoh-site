def __init__(self, dim: int = 8):  # Smaller for sim
    self.dim = dim
    self.graph = nx.hypercube_graph(dim)

def project(self, point: np.ndarray) -> np.ndarray:
    from sklearn.decomposition import PCA
    pca = PCA(n_components=3)
    return pca.fit_transform(point.reshape(1, -1))[0]

