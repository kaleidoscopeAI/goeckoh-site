import numpy as np
from sklearn.metrics import pairwise_distances

class PhiCalculator:
    """Compute multi-scale Phi (Eq.5)"""
    def compute_phi(self, node, nodes=None):
        """Approximates as information integration among neighbors.
        node: Node instance
        nodes: list of connected Node instances
        """
        if not nodes:
            return 0.1  # minimal base Phi

        embeddings = np.array([n.embedding for n in nodes] + [node.embedding])
        cov = np.cov(embeddings, rowvar=False)
        eigvals = np.linalg.eigvalsh(cov)
        phi = np.sum(eigvals**2)
        return phi
