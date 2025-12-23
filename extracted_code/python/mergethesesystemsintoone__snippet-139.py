def mutate(self):
    """Evolve pattern recognition capability"""
    if np.random.random() < self.adaptation_rate:
        if len(self.sequence) > 3:
            # Combine existing patterns
            idx1, idx2 = np.random.choice(len(self.sequence), 2, replace=False)
            new_pattern = self.sequence[idx1][:2] + self.sequence[idx2][2:]
            self.sequence.append(new_pattern)
            self.mutations += 1
