"""Dynamic thresholds that adjust based on performance"""
base_thresholds: Dict[str, float] = field(default_factory=lambda: {
    'survival': 0.3,
    'growth': 0.5,
    'replication': 0.7,
    'teaching': 0.8
})
adaptation_rate: float = 0.05
history: Dict[str, List[float]] = field(default_factory=lambda: {
    'survival': [],
    'growth': [],
    'replication': [],
    'teaching': []
})

def adapt_thresholds(self, performance_history: List[float]):
    """Adjust thresholds based on performance"""
    if not performance_history:
        return

    recent_performance = np.mean(performance_history[-10:])
    for mode, threshold in self.base_thresholds.items():
        if recent_performance > threshold + 0.1:
            # Increase threshold if consistently exceeding it
            self.base_thresholds[mode] += self.adaptation_rate
        elif recent_performance < threshold - 0.1:
            # Decrease threshold if consistently falling short
            self.base_thresholds[mode] = max(
                0.1,
                self.base_thresholds[mode] - self.adaptation_rate
            )

        self.history[mode].append(self.base_thresholds[mode])

