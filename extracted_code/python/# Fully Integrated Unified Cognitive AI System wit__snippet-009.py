from numba import njit

@njit
def update_node_vector_fast(vector, temperature, tension, noise_std=0.01):
    influence = np.tanh(temperature - tension)
    noise = np.random.randn(vector.size) * noise_std
    vector += influence * noise
    norm = np.linalg.norm(vector)
    if norm > 0:
        vector /= norm
    return vector

class CognitiveCube:
    # ...
    def iterate(self):
        self.env.fluctuate()
        for node in self.nodes:
            node.vector = update_node_vector_fast(node.vector, self.env.temperature, node.tension)
            # Additional node updates
            node.normalize()
