noise_std_initial = 0.1
decay_rate = 1.0

for k in range(steps):
    noise = np.random.normal(0, noise_std_initial * np.exp(-decay_rate * k * dt), size=2)
    z = z + dt * f(z) + noise
    dist = np.linalg.norm(x - z)
    distances.append(dist)
