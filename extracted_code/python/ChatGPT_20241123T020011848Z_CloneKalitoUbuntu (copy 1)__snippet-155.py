class ObjectiveWeights:
    """Weights for objective score calculation"""
    growth: float = 0.3      # Weight for growth factor
    efficiency: float = 0.3  # Weight for energy efficiency
    knowledge: float = 0.4   # Weight for knowledge gain
    
    def validate(self):
        """Ensure weights sum to 1.0"""
        total = self.growth + self.efficiency + self.knowledge
        if abs(total - 1.0) > 0.001:
            scale = 1.0 / total
            self.growth *= scale
            self.efficiency *= scale
            self.knowledge *= scale

