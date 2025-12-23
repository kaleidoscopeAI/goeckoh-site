class DynamicWeights:
    """Adaptive weights that adjust based on conditions"""
    growth: float = 0.3
    efficiency: float = 0.3
    knowledge: float = 0.4
    adaptation_rate: float = 0.1
    history: Dict[str, List[float]] = field(default_factory=lambda: {
        'growth': [],
        'efficiency': [],
        'knowledge': []
    })

    def adapt_to_conditions(self, environment_state: Dict, node_state: Dict):
        """Dynamically adjust weights based on conditions"""
        # Calculate pressure factors
        energy_pressure = 1 - (node_state['energy'] / node_state['max_energy'])
        growth_pressure = 1 - len(node_state['connections']) / 10
        knowledge_pressure = 1 - len(node_state['knowledge_base']) / 100

        # Adjust weights based on pressures
        self.efficiency = min(0.6, self.efficiency + energy_pressure * self.adaptation_rate)
        self.growth = min(0.6, self.growth + growth_pressure * self.adaptation_rate)
        self.knowledge = min(0.6, self.knowledge + knowledge_pressure * self.adaptation_rate)

        # Normalize weights
        total = self.growth + self.efficiency + self.knowledge
        self.growth /= total
        self.efficiency /= total
        self.knowledge /= total

        # Record history
        self._record_weights()

    def _record_weights(self):
        """Record weight history for analysis"""
        self.history['growth'].append(self.growth)
        self.history['efficiency'].append(self.efficiency)
        self.history['knowledge'].append(self.knowledge)

