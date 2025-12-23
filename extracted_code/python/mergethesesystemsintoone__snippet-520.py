text_patterns: List[PatternStrand] = field(default_factory=list)
visual_patterns: List[VisualStrand] = field(default_factory=list)
mutation_rate: float = 0.01
generation: int = 0

def replicate(self) -> 'KnowledgeDNA':
    new_dna = KnowledgeDNA(mutation_rate=self.mutation_rate, generation=self.generation + 1)
    for p in self.text_patterns:
        new_p = PatternStrand(p.sequence[:], p.strength, p.adaptation_rate)
        new_p.mutate()
        new_dna.text_patterns.append(new_p)
    for v in self.visual_patterns:
        new_v = VisualStrand(v.feature_patterns.copy(), v.mutation_rate)
        new_v.evolve(np.random.randn(10))  # Real evolution with random features
        new_dna.visual_patterns.append(new_v)
    return new_dna

