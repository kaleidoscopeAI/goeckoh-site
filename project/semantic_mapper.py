import numpy as np
from sklearn.preprocessing import normalize

class HolographicSemanticMapper:
    """Implements Eq.3 from Axiom V:
    x_i = F_base(E_i) + F_harmonic(E_i) + F_context(E_i)
    Maps probabilistic genomes to a geometric semantic space.
    """
    def __init__(self, genome_length=128, embedding_dim=64, seed=42):
        self.genome_length = genome_length
        self.embedding_dim = embedding_dim
        self.rng = np.random.default_rng(seed)
        # Precompute random projection matrices for base and harmonic terms
        self.W_base = self.rng.normal(0, 1, (embedding_dim, genome_length))
        self.W_harmonic = self.rng.normal(0, 1, (embedding_dim, genome_length))

    def F_base(self, E):
        """Linear projection of genome bits to embedding space"""
        return self.W_base @ E

    def F_harmonic(self, E):
        """Harmonic encoding (sin/cos) for periodicity/phase info"""
        indices = np.arange(len(E))
        return np.sum(np.stack([np.sin(E * indices), np.cos(E * indices)]), axis=0)[:self.embedding_dim]

    def F_context(self, E, neighbors=None):
        """Contextual pull from neighbor embeddings"""
        if neighbors is None or len(neighbors) == 0:
            return np.zeros(self.embedding_dim)
        return np.mean(neighbors, axis=0)

    def embed(self, E, neighbors=None):
        """Full holographic embedding: base + harmonic + context
        E: 1D array of probabilistic genome (length genome_length)
        neighbors: list of 1D embeddings
        """
        base = self.F_base(E)
        harmonic = self.F_harmonic(E)
        context = self.F_context(E, neighbors)
        embedding = base + harmonic + context
        return normalize(embedding.reshape(1, -1))[0]  # L2 normalize
