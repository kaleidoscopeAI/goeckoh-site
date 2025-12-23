"""Generates Hamiltonians for quantum evolution"""

def __init__(self, dimension: int = HILBERT_SPACE_DIM):
    self.dimension = dimension
    self.observable_gen = ObservableGenerator(dimension)

def get_base_hamiltonian(self) -> np.ndarray:
    """Create a basic Hamiltonian for time evolution"""
    # Start with energy observable as the base
    H_base = self.observable_gen.get_energy_observable()
    return H_base

def get_network_hamiltonian(self, network_graph: nx.Graph, node_mapping: Dict[str, int]) -> np.ndarray:
    """Create a Hamiltonian based on network topology"""
    H = np.zeros((self.dimension, self.dimension), dtype=complex)

    # Add interaction terms for connected nodes
    for node1, node2, attrs in network_graph.edges(data=True):
        if node1 in node_mapping and node2 in node_mapping:
            # Get indices in the Hamiltonian
            i = node_mapping[node1] % self.dimension
            j = node_mapping[node2] % self.dimension

            # Get interaction strength (default to 0.5)
            strength = attrs.get('weight', 0.5)

            # Add interaction term
            phase = np.exp(1j * np.pi * (i*j) / self.dimension)
            interaction = strength * phase
            H[i, j] += interaction
            H[j, i] += np.conjugate(interaction)

    # Add diagonal terms
    for node, idx in node_mapping.items():
        i = idx % self.dimension
        # Node degree affects diagonal energy
        degree = network_graph.degree(node)
        H[i, i] = 1.0 + 0.1 * degree

    # Ensure Hermitian (H = H†)
    H = 0.5 * (H + H.conj().T)

    return H

def get_evolution_hamiltonian(self, network_graph: nx.Graph, node_mapping: Dict[str, int]) -> np.ndarray:
    """Create a complete Hamiltonian for time evolution"""
    # Combine base and network Hamiltonians
    H_base = self.get_base_hamiltonian()
    H_network = self.get_network_hamiltonian(network_graph, node_mapping)

    # Weight between isolation and network effects
    alpha = 0.7  # Favor network effects
    H = (1.0 - alpha) * H_base + alpha * H_network

    # Ensure Hermitian
    H = 0.5 * (H + H.conj().T)

    return H

