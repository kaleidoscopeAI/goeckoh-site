import numpy as np

class HamiltonianController:
    """Compute total Hamiltonian Eq.4"""
    def compute(self, node, nodes=None):
        H_cognitive = node.similarity(np.mean([n.embedding for n in nodes], axis=0)) if nodes else 0
        H_safety = np.sum((node.E - 0.5)**2)
        H_consciousness = node.energy
        H_evolution = np.var([n.energy for n in nodes]) if nodes else 0
        H_total = H_cognitive + H_safety + H_consciousness + H_evolution
        return H_total
