class KnowledgeDNA:
    """Complex DNA structure for knowledge representation"""
    text_patterns: List[PatternStrand] = field(default_factory=list)
    visual_patterns: List[VisualStrand] = field(default_factory=list)
    connection_strength: Dict[Tuple[str, str], float] = field(default_factory=dict)
    mutation_rate: float = 0.01
    generation: int = 0

    def replicate(self):
        """Create new DNA strand with possible mutations"""
        new_dna = KnowledgeDNA(mutation_rate=self.mutation_rate)
        new_dna.generation = self.generation + 1

        # Replicate text patterns with possible mutations
        for pattern in self.text_patterns:
            new_pattern = PatternStrand(
                sequence=pattern.sequence.copy(),
                strength=pattern.strength,
                adaptation_rate=pattern.adaptation_rate
            )
            if np.random.random() < self.mutation_rate:
                new_pattern.mutate()
            new_dna.text_patterns.append(new_pattern)

        # Replicate visual patterns with adaptation
        for pattern in self.visual_patterns:
            new_visual = VisualStrand()
            for key, features in pattern.feature_patterns.items():
                noise = np.random.normal(0, self.mutation_rate, features.shape)
                new_visual.feature_patterns[key] = features + noise
            new_dna.visual_patterns.append(new_visual)

        return new_dna
