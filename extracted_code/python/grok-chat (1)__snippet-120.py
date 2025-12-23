class DecisionAllocation:
    def allocate(self, node_id, cpu=0.5, mem=0.3):
        score = 0.4 * cpu + 0.3 * mem
        return score > 0.5  # Allocate if high

