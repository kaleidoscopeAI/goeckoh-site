# ...
def step(self) -> None:
    T = float(self.anneal_fn(self.t)) # Temperature decreases over time.
    node = random.choice(list(self.state.E.keys()))
    bit_idx = random.randrange(int(np.asarray(self.state.E[node]).size))

    # Efficiently calculates energy change for just one bit flip.
    delta = self.ham.delta_energy_for_bitflip(self.state, node, bit_idx)

    # Accepts the flip if it lowers energy OR randomly based on temperature.
    accept_prob = min(1.0, math.exp(-delta / (T + 1e-12)))
    if random.random() < accept_prob:
        # ... (applies the bit flip) ...

