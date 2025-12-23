class PatternStrand:
    sequence: List[str] = field(default_factory=list)
    strength: float = 0.0
    adaptation_rate: float = 0.1

    def mutate(self):
        if random.random() < self.adaptation_rate and self.sequence:
            idx = random.randint(0, len(self.sequence)-1)
            self.sequence[idx] = self.sequence[idx][::-1]  # Reverse as mutation (real logic)
            self.strength += random.uniform(-0.05, 0.05)

