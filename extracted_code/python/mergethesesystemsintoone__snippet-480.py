class CognitiveEngine:
    def __init__(self, input_dim: int):
        self.weights = np.random.randn(input_dim, 3) * 0.1

    def forward(self, state: np.ndarray) -> np.ndarray:
        z = np.tanh(state.dot(self.weights))
        return 1 / (1 + np.exp(-z))

