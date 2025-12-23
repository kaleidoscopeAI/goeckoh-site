class QuantumInformationState:
    """Quantum-inspired information state for documents"""
    amplitude_0: complex = complex(1/np.sqrt(2), 0)  # |0⟩ state
    amplitude_1: complex = complex(0, 1/np.sqrt(2))  # |1⟩ state
    phase: float = 0.0
    entanglement_degree: float = 0.0
    
    def measure_probability(self) -> float:
        """Measure quantum information probability"""
        return abs(self.amplitude_1)**2

