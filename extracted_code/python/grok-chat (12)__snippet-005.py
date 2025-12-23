class CrystallineHeart:
    def __init__(self, n_nodes=128, alpha=1.0, beta=0.5, gamma=0.2):
        self.n = n_nodes
        self.E = np.zeros(n_nodes, dtype=np.float32)
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.W = np.random.rand(n_nodes, n_nodes).astype(np.float32) * 0.5

    def update_and_get_gcl(self, input_val):
        try:
            mean_E = np.mean(self.E)
            dE = self.alpha * input_val - self.beta * self.E + self.gamma * np.dot(self.W, (self.E - mean_E))
            self.E += dE * 0.01
            self.E = np.tanh(self.E)
            gcl = np.mean(np.abs(self.E))
            A = -self.beta * np.eye(self.n, dtype=np.float32) + self.gamma * self.W
            eigs = np.linalg.eigvals(A)
            if np.any(np.real(eigs) >= 0):
                self.W *= 0.9
            return gcl
        except Exception as e:
            logger.error(f"GCL error: {e}")
            return 0.5

