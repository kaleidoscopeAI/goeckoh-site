class EnergyFlow:
    def __init__(self):
        self.node_energy: Dict[str, float] = {}
        self.pq: List[Tuple[float, str]] = []  # (energy, node_id)

    def add_node(self, node_id: str, energy: float = 100.0):
        self.node_energy[node_id] = energy
        heapq.heappush(self.pq, (energy, node_id))

    def redistribute(self, threshold: float = 50.0):
        low = [n for n, e in self.node_energy.items() if e < threshold]
        high = [n for n, e in self.node_energy.items() if e > threshold]
        if low and high:
            for h in high:
                donation = min(self.node_energy[h] - threshold, threshold - sum(self.node_energy[l] for l in low) / len(low))
                for l in low:
                    self.node_energy[l] += donation / len(low)
                self.node_energy[h] -= donation

