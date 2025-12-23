"""Robust quantum state with pure NumPy implementation"""
hamiltonian: np.ndarray = field(default_factory=lambda: np.eye(3))
wavefunction: np.ndarray = field(default_factory=lambda: np.ones(3)/np.sqrt(3))
energy: float = 0.0
correlation_length: float = 5.0
criticality_index: float = 1.0

def evolve_pure_numpy(self, dt: float = 0.01):
    """Pure NumPy quantum evolution without scipy dependency"""
    # Taylor series expansion of matrix exponential
    # exp(-iHdt) ≈ I - iHdt - (Hdt)²/2 + i(Hdt)³/6
    Hdt = -1j * self.hamiltonian * dt

    # First few terms of Taylor series - ensure complex dtype
    evolution_matrix = np.eye(self.hamiltonian.shape[0], dtype=complex)
    evolution_matrix += Hdt  # First order
    evolution_matrix += Hdt @ Hdt / 2  # Second order
    evolution_matrix += Hdt @ Hdt @ Hdt / 6  # Third order

    self.wavefunction = evolution_matrix @ self.wavefunction
    self.wavefunction /= np.linalg.norm(self.wavefunction)
    self.energy = np.real(self.wavefunction.conj().T @ self.hamiltonian @ self.wavefunction)

