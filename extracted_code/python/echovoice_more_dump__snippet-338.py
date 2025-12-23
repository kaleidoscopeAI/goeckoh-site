class ObservableGenerator:
    """Generates quantum observables for different measurements"""
    
    def __init__(self, dimension: int = HILBERT_SPACE_DIM):
        self.dimension = dimension
        self.cache = {}  # Cache generated observables
    
    def get_random_observable(self) -> np.ndarray:
        """Generate a random Hermitian observable"""
        # Create random Hermitian (H = H†)
        H = np.random.normal(0, 1, (self.dimension, self.dimension)) + \
            1j * np.random.normal(0, 1, (self.dimension, self.dimension))
        H = 0.5 * (H + H.conj().T)  # Ensure Hermitian
        return H
    
    def get_energy_observable(self) -> np.ndarray:
        """Get observable corresponding to energy measurement"""
        if "energy" in self.cache:
            return self.cache["energy"]
        
        # Create an energy observable with increasing eigenvalues
        diagonal = np.arange(self.dimension) / self.dimension
        H = np.diag(diagonal)
        
        # Add small off-diagonal elements for "interactions"
        for i in range(self.dimension-1):
            H[i, i+1] = H[i+1, i] = 0.1 / self.dimension
        
        self.cache["energy"] = H
        return H
    
    def get_coherence_observable(self) -> np.ndarray:
        """Get observable to measure quantum coherence"""
        if "coherence" in self.cache:
            return self.cache["coherence"]
        
        # Create a sparse matrix with high eigenvalues for coherent states
        H = np.zeros((self.dimension, self.dimension), dtype=complex)
        
        # Add elements that favor superposition states
        for i in range(self.dimension):
            for j in range(i+1, self.dimension):
                phase = np.exp(1j * 2 * np.pi * (i+j) / self.dimension)
                H[i, j] = 0.1 * phase
                H[j, i] = 0.1 * np.conjugate(phase)
        
        # Add diagonal
        for i in range(self.dimension):
            H[i, i] = 1.0 - 0.5 * (i / self.dimension)
        
        self.cache["coherence"] = H
        return H
    
    def get_entanglement_observable(self, node_id: str, target_id: str) -> np.ndarray:
        """Get observable to measure entanglement between two nodes"""
        cache_key = f"entanglement_{hash(node_id)}_{hash(target_id)}"
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        # Create entanglement observable based on node IDs
        H = np.zeros((self.dimension, self.dimension), dtype=complex)
        
        # Use hash of node IDs to create a unique observable
        seed = int(hashlib.md5((node_id + target_id).encode()).hexdigest(), 16) % 10000
        np.random.seed(seed)
        
        # Create Bell-like measurement projectors
        # Simplified - in a real system, would depend on actual entangled states
        for i in range(0, self.dimension // 2):
            j = self.dimension - i - 1
            # Create projector onto maximally entangled state |i,j⟩ + |j,i⟩
            proj = np.zeros((self.dimension, self.dimension), dtype=complex)
            
            # Set elements for |i⟩⟨j| and |j⟩⟨i|
            proj[i, j] = proj[j, i] = 1.0 / np.sqrt(2)
            
            # Add to Hamiltonian with random strength
            strength = 0.5 + 0.5 * np.random.random()
            H += strength * proj
        
        # Ensure Hermitian
        H = 0.5 * (H + H.conj().T)
        
        self.cache[cache_key] = H
        return H
    
    def get_pauli_operators(self) -> Dict[str, np.ndarray]:
        """Get generalized Pauli operators for the Hilbert space"""
        if "pauli" in self.cache:
            return self.cache["pauli"]
        
        # For qubits, we need log2(dimension) qubits
        n_qubits = int(np.ceil(np.log2(self.dimension)))
        actual_dim = 2**n_qubits
        
        # Pauli matrices
        sigma_x = np.array([[0, 1], [1, 0]])
        sigma_y = np.array([[0, -1j], [1j, 0]])
        sigma_z = np.array([[1, 0], [0, -1]])
        identity = np.eye(2)
        
        operators = {}
        
        # Create Pauli operators for each qubit
        for i in range(n_qubits):
            # Create X operator for qubit i
            X_i = np.array(1.0)
            for j in range(n_qubits):
                if j == i:
                    X_i = np.kron(X_i, sigma_x)
                else:
                    X_i = np.kron(X_i, identity)
            operators[f"X_{i}"] = X_i[:self.dimension, :self.dimension]  # Truncate if needed
            
            # Create Z operator for qubit i
            Z_i = np.array(1.0)
            for j in range(n_qubits):
                if j == i:
                    Z_i = np.kron(Z_i, sigma_z)
                else:
                    Z_i = np.kron(Z_i, identity)
            operators[f"Z_{i}"] = Z_i[:self.dimension, :self.dimension]  # Truncate if needed
            
            # Create Y operator for qubit i
            Y_i = np.array(1.0)
            for j in range(n_qubits):
                if j == i:
                    Y_i = np.kron(Y_i, sigma_y)
                else:
                    Y_i = np.kron(Y_i, identity)
            operators[f"Y_{i}"] = Y_i[:self.dimension, :self.dimension]  # Truncate if needed
        
        self.cache["pauli"] = operators
        return operators

class HamiltonianGenerator:
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

