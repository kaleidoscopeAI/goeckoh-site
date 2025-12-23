class QuantumState:
    """Represents a quantum state in the Hilbert space"""
    
    dimension: int = HILBERT_SPACE_DIM
    state: Optional[np.ndarray] = None
    fidelity: float = 1.0
    creation_time: float = field(default_factory=time.time)
    collapse_status: WavefunctionCollapse = WavefunctionCollapse.COHERENT
    entangled_with: Set[str] = field(default_factory=set)
    
    def __post_init__(self):
        """Initialize state if not provided"""
        if self.state is None:
            # Create random pure state
            state_vector = np.random.normal(0, 1, self.dimension) + 1j * np.random.normal(0, 1, self.dimension)
            self.state = state_vector / np.linalg.norm(state_vector)
    
    def evolve(self, hamiltonian: np.ndarray, dt: float) -> None:
        """Evolve quantum state according to Schrödinger equation"""
        # U = exp(-i*H*dt)
        evolution_operator = np.exp(-1j * hamiltonian * dt)
        self.state = evolution_operator @ self.state
        # Renormalize to handle numerical errors
        self.state = self.state / np.linalg.norm(self.state)
    
    def apply_noise(self, dt: float) -> None:
        """Apply decoherence to the quantum state"""
        elapsed_time = time.time() - self.creation_time
        decoherence = np.exp(-QUANTUM_DECOHERENCE_RATE * elapsed_time)
        
        # Mix with random state based on decoherence
        if decoherence < 1.0:
            random_state = np.random.normal(0, 1, self.dimension) + 1j * np.random.normal(0, 1, self.dimension)
            random_state = random_state / np.linalg.norm(random_state)
            
            # Apply partial decoherence
            self.state = decoherence * self.state + np.sqrt(1 - decoherence**2) * random_state
            self.state = self.state / np.linalg.norm(self.state)
            
            # Update fidelity
            self.fidelity = decoherence
    
    def measure(self, observable: np.ndarray) -> float:
        """Measure an observable on the quantum state"""
        # <ψ|O|ψ>
        expectation = np.real(np.conjugate(self.state) @ observable @ self.state)
        return expectation
    
    def measure_with_collapse(self, observable: np.ndarray) -> Tuple[float, 'QuantumState']:
        """Measure with wavefunction collapse, returning result and new state"""
        # Calculate expectation value
        expectation = self.measure(observable)
        
        # Perform eigendecomposition of the observable
        eigenvalues, eigenvectors = np.linalg.eigh(observable)
        
        # Calculate probabilities based on Born rule
        probabilities = np.abs(np.conjugate(eigenvectors.T) @ self.state)**2
        
        # Choose an eigenvalue based on probabilities
        result_idx = np.random.choice(len(eigenvalues), p=probabilities.real)
        result = eigenvalues[result_idx]
        
        # State collapses to corresponding eigenvector
        new_state = QuantumState(dimension=self.dimension)
        new_state.state = eigenvectors[:, result_idx]
        new_state.collapse_status = WavefunctionCollapse.CLASSICAL
        
        return result, new_state
    
    def entangle_with(self, other_id: str) -> None:
        """Mark this state as entangled with another node"""
        self.entangled_with.add(other_id)
        if len(self.entangled_with) > 0:
            self.collapse_status = WavefunctionCollapse.ENTANGLED
    
    def compute_entropy(self) -> float:
        """Calculate the von Neumann entropy of the state"""
        # S = -Tr(ρ ln ρ) where ρ is the density matrix |ψ⟩⟨ψ|
        # For pure states, entropy is 0. For mixed states, it's positive.
        if self.fidelity > 0.99:  # Pure state
            return 0.0
        
        # Create density matrix ρ = |ψ⟩⟨ψ|
        density_matrix = np.outer(self.state, np.conjugate(self.state))
        
        # Calculate eigenvalues of density matrix
        eigenvalues = np.linalg.eigvalsh(density_matrix)
        
        # Keep only positive eigenvalues (numerical issues may give tiny negative values)
        eigenvalues = eigenvalues[eigenvalues > 1e-10]
        
        # Calculate entropy: S = -Σ λ_i ln λ_i
        entropy = -np.sum(eigenvalues * np.log(eigenvalues))
        return float(entropy)
    
    def serialize(self) -> Dict[str, Any]:
        """Serialize the quantum state for transmission"""
        return {
            "real": self.state.real.tolist(),
            "imag": self.state.imag.tolist(),
            "fidelity": self.fidelity,
            "creation_time": self.creation_time,
            "collapse_status": self.collapse_status.name,
            "entangled_with": list(self.entangled_with)
        }
    
    @classmethod
    def deserialize(cls, data: Dict[str, Any]) -> 'QuantumState':
        """Deserialize a quantum state from received data"""
        state = cls(dimension=len(data["real"]))
        state.state = np.array(data["real"]) + 1j * np.array(data["imag"])
        state.fidelity = data["fidelity"]
        state.creation_time = data["creation_time"]
        state.collapse_status = WavefunctionCollapse[data["collapse_status"]]
        state.entangled_with = set(data["entangled_with"])
        return state

