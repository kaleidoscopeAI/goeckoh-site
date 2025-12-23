\"\"\"DNA-like structure for pattern recognition\"\"\"
sequence: List[str] = field(default_factory=list)
strength: float = 0.0
mutations: int = 0
activation_threshold: float = 0.5
adaptation_rate: float = 0.1

def mutate(self):
    \"\"\"Evolve pattern recognition capability\"\"\"
    if np.random.random() < self.adaptation_rate:
        if len(self.sequence) > 3:
            # Combine existing patterns
            idx1, idx2 = np.random.choice(len(self.sequence), 2, replace=False)
            new_pattern = self.sequence[idx1][:2] + self.sequence[idx2][2:]
            self.sequence.append(new_pattern)
            self.mutations += 1

