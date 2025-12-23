class VisualStrand:
    feature_patterns: Dict[str, np.ndarray] = field(default_factory=dict)
    mutation_rate: float = 0.1

    def evolve(self, new_features: np.ndarray):
        key = list(self.feature_patterns.keys())[0] if self.feature_patterns else "default"
        if key not in self.feature_patterns:
            self.feature_patterns[key] = new_features
        else:
            noise = np.random.normal(0, self.mutation_rate, new_features.shape)
            self.feature_patterns[key] = 0.8 * self.feature_patterns[key] + 0.2 * new_features + noise

