import numpy as np
import matplotlib.pyplot as plt

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

for k in range(timesteps):
    x1 = T(x1)
    x2 = T(x2)
    dist = np.linalg.norm(x1 - x2)
    distances.append(dist)

plt.plot(distances, 'o-')
plt.yscale('log')
plt.xlabel('Iteration k')
plt.ylabel('Distance between trajectories (log scale)')
plt.title('Exponential contraction of trajectories in RNN')
plt.grid(True)
plt.show()
