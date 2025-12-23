def __init__(self):
    self.node_energy: Dict[str, float] = {}
    self.pq: List[Tuple[float, str]] = []  # (-energy for max-heap donors, but use min-heap for low)

def add_node(self, node_id: str, energy: float = 100.0):
    if node_id not in self.node_energy:
        self.node_energy[node_id] = energy
        heapq.heappush(self.pq, (energy, node_id))

def redistribute(self, threshold: float = 50.0):
    # Real logic: Sort low/high
    low = sorted([ (e, n) for n, e in self.node_energy.items() if e < threshold ])
    high = sorted([ (e, n) for n, e in self.node_energy.items() if e > threshold ], reverse=True)
    total_deficit = sum(threshold - e for e, _ in low)
    for he, hn in high:
        donation = min(he - threshold, total_deficit)
        he -= donation
        total_deficit -= donation
        for i in range(len(low)):
            le, ln = low[i]
            needed = threshold - le
            transfer = min(needed, donation / len(low))
            low[i] = (le + transfer, ln)
        if total_deficit <= 0:
            break
    # Update dict
    for e, n in low + high:
        self.node_energy[n] = e
    self.pq = [(self.node_energy[n], n) for n in self.node_energy]
    heapq.heapify(self.pq)

