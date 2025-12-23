import numpy as np
import matplotlib.pyplot as plt

# Parameters
W = np.array([[0.3, 0.2],
              [0.1, 0.4]])
b = np.array([0.1, -0.1])
timesteps = 30

def activation(x):
    return np.tanh(x)

def T(x):
    return np.dot(W, activation(x)) + b

# Initial states
x1 = np.array([0.0, 0.0])
x2 = np.array([1.0, 1.0])

distances = []

# Iterate and record distances
for _ in range(timesteps):
    x1 = T(x1)
    x2 = T(x2)
    dist = np.linalg.norm(x1 - x2)
    distances.append(dist)

# Plot distance decay
plt.figure(figsize=(8,4))
plt.plot(distances, marker='o')
plt.yscale('log')
plt.xlabel('Iteration k')
plt.ylabel('Distance ||x1_k - x2_k|| (log scale)')
plt.title('Exponential decay in distance between trajectories (Contraction)')
plt.grid(True)
plt.show()
