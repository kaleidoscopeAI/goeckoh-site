class EnergyFlow:
    def __init__(self):
        self.node_energy: Dict[str, float] = {}
        self.pq: List[Tuple[float, str]] = []

    def add_node_batch(self, nodes: List[Tuple[str, float]]):
        for n, e in nodes:
            if n not in self.node_energy:
                self.node_energy[n] = e
                heapq.heappush(self.pq, (e, n))

    def redistribute(self, threshold: float = 50.0):
        while self.pq and self.pq[0][0] < threshold:
            low_e, low_n = heapq.heappop(self.pq)
            if low_e != self.node_energy.get(low_n, float('inf')): continue
            high = sorted([(self.node_energy[n], n) for n in self.node_energy if self.node_energy[n] > threshold], reverse=True)[:BATCH_SIZE]
            deficit = threshold - low_e
            for high_e, high_n in high:
                donation = min(high_e - threshold, deficit)
                self.node_energy[high_n] -= donation
                deficit -= donation
                heapq.heappush(self.pq, (self.node_energy[high_n], high_n))
                if deficit <= 0: break
            self.node_energy[low_n] = threshold - deficit
            heapq.heappush(self.pq, (self.node_energy[low_n], low_n))

