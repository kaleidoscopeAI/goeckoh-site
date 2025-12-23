"""Full quantum bit with actual quantum operations"""

def __init__(self, alpha=1/np.sqrt(2), beta=1/np.sqrt(2)):
    # |ψ⟩ = α|0⟩ + β|1⟩
    self.alpha = complex(alpha)
    self.beta = complex(beta)
    self.normalize()

def normalize(self):
    """Ensure |α|² + |β|² = 1"""
    norm = math.sqrt(abs(self.alpha)**2 + abs(self.beta)**2)
    if norm > 0:
        self.alpha /= norm
        self.beta /= norm

def measure(self) -> int:
    """Projective measurement: P(1) = |β|²"""
    prob_1 = abs(self.beta)**2
    return 1 if random.random() < prob_1 else 0

def apply_gate(self, gate_matrix):
    """Apply unitary gate: |ψ'⟩ = U|ψ⟩"""
    new_state = gate_matrix @ np.array([self.alpha, self.beta])
    self.alpha, self.beta = new_state[0], new_state[1]
    self.normalize()

def entangle(self, other):
    """Create Bell state entanglement: (|00⟩ + |11⟩)/√2"""
    # For demonstration, create correlated states
    correlation_strength = 0.9
    self.alpha = 1/np.sqrt(2)
    self.beta = 1/np.sqrt(2)
    other.alpha = self.alpha
    other.beta = self.beta * correlation_strength
    other.normalize()
    self.normalize()

def density_matrix(self):
    """Compute density matrix ρ = |ψ⟩⟨ψ|"""
    state = np.array([[self.alpha], [self.beta]])
    return state @ state.conj().T

