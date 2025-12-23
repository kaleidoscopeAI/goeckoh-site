"""Simulates quantum operations"""

def __init__(self, qubits: int = 8):
    self.n_qubits = qubits
    self.state_vector = np.zeros(2**qubits, dtype=complex)
    self.state_vector[0] = 1.0
    logger.info(f"Initialized {qubits}-qubit quantum simulator")

def apply_hadamard(self, target: int):
    """Apply Hadamard gate to create superposition."""
    if target >= self.n_qubits:
        raise ValueError(f"Target qubit {target} out of range")
    h = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
    for i in range(0, 2**self.n_qubits, 2**(target+1)):
        for j in range(2**target):
            idx1, idx2 = i + j, i + j + 2**target
            self.state_vector[idx1], self.state_vector[idx2] = (
                h[0, 0] * self.state_vector[idx1] + h[0, 1] * self.state_vector[idx2],
                h[1, 0] * self.state_vector[idx1] + h[1, 1] * self.state_vector[idx2],
            )

def measure(self) -> List[int]:
    """Measure all qubits and collapse the state."""
    probabilities = np.abs(self.state_vector)**2
    outcome = np.random.choice(2**self.n_qubits, p=probabilities)
    self.state_vector = np.zeros(2**self.n_qubits, dtype=complex)
    self.state_vector[outcome] = 1.0
    return [int(b) for b in format(outcome, f'0{self.n_qubits}b')]

