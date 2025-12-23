from numba import njit
import numpy as np

@njit
def update_node_vector(vector, env_temperature, tension, noise_std=0.01):
    influence = np.tanh(env_temperature - tension)
    noise = np.random.randn(vector.shape[0]) * noise_std
    vector = vector + influence * noise
    norm = np.linalg.norm(vector)
    if norm > 0:
        vector = vector / norm
    return vector

# Example usage updating node vectors in batch to accelerate iterations
for node in self.nodes:
    node.vector = update_node_vector(node.vector, self.env.temperature, node.tension)
