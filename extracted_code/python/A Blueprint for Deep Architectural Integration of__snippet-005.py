dt = 0.01
T_final = 5
steps = int(T_final / dt)

W = 0.5 * np.eye(2)

def f(x):
    return -x + np.dot(W, np.tanh(x))

x = np.array([0.0, 1.0])
z = np.array([1.0, -1.0])

distances = []

for _ in range(steps):
    x = x + dt * f(x)
    z = z + dt * f(z)
    dist = np.linalg.norm(x - z)
    distances.append(dist)

plt.plot(np.linspace(0, T_final, steps), distances)
plt.xlabel('Time')
plt.ylabel('Distance between trajectories')
plt.title('Exponential contraction in continuous-time system')
plt.show()
