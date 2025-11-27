import numpy as np

class PredictiveSafetySystem:
    """Monte Carlo future-state sampling + intervention planning"""
    def simulate_future(self, node, steps=10):
        energies = [node.energy]
        for _ in range(steps):
            delta = np.random.normal(0, 0.01)
            energies.append(max(0, energies[-1] + delta))
        return energies

    def intervene(self, node):
        if node.energy < 0.1:
            node.energy = 0.5  # simple emergency reset
            return True
        return False
