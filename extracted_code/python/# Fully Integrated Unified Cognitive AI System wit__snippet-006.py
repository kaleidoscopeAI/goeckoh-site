from numba import njit
import numpy as np

@njit
def update_node_vector_fast(vector, temperature, tension, noise_std=0.01):
    influence = np.tanh(temperature - tension)
    noise = np.random.randn(vector.size) * noise_std
    vector += influence * noise
    norm = np.linalg.norm(vector)
    if norm > 0:
        vector /= norm
    return vector

# Use this in your node update loop to speed up vector updates:
for node in self.nodes:
    node.vector = update_node_vector_fast(node.vector, self.env.temperature, node.tension)
