class Hypercube:
    def __init__(self, dim: int = 16):
        self.dim = dim
        self.graph = nx.hypercube_graph(dim)

    @lru_cache(maxsize=1024)
    def project_batch(self, points: Tuple[np.ndarray]) -> List[np.ndarray]:
        from sklearn.decomposition import PCA
        pca = PCA(n_components=3)
        points_array = np.vstack(points)
        return pca
        
        #!/usr/bin/env python3
