"""
(C) Implements Dynamic Parameter Scaling for w, c1, c2.

Args:
    t (int): Current iteration number.
    T_max (int): Maximum iteration number.
    delta_h (float): CA Shannon Entropy (from CellularChaosGenerator).
    w_bounds (tuple): (w_min, w_max)
    c_bounds (tuple): (c_min, c_max)

Returns:
    dict: {'w': w, 'c1': c1, 'c2': c2}
"""
# Normalized time (linear decay factor)
t_norm = t / T_max

# Inertia Weight (w): Decays from w_max to w_min, boosted by Chaos Entropy
w_min, w_max = w_bounds
w_base = w_max - (w_max - w_min) * t_norm # Linear decay

# Entropy Scaling Factor: Boost inertia when CA is highly chaotic (delta_h is high)
# Use a smooth scaling function, e.g., logistic sigmoid or simple quadratic.
# Assume delta_h is normalized or bounded (e.g., max ~10 for a 100x100 grid with 2 states).
chaos_boost = 1.0 + (delta_h / 5.0)**2 # Simple scaling (tuned for reliability)

w = np.clip(w_base * chaos_boost, w_min, w_max * 1.5) # Allow temporary overshoot

# Cognitive (c1) and Social (c2) components: Complementary scheduling
# c1 (Cognitive/Individual): Decays over time (Exploitation focus shifts from individual)
# c2 (Social/Global): Increases over time (Exploitation focus shifts to global/g_best)
c_min, c_max = c_bounds
c1 = c_max - (c_max - c_min) * t_norm
c2 = c_min + (c_max - c_min) * t_norm

# Add minor perturbation based on entropy to c1/c2 for self-adaptive exploration 
# (high entropy = more randomness in learning rates)
c1 += (np.random.rand() * 0.1) * delta_h / 5.0
c2 += (np.random.rand() * 0.1) * delta_h / 5.0

return {'w': c1, 'c1': c1, 'c2': c2}

