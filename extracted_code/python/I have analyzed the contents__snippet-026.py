"""Background process that 'dreams' and grows based on system events."""
def __init__(self):
    self.nodes = [OrganicNode("root", NodeDNA([0.1, 0.5, 0.9]))]
    self.lock = threading.Lock()
    self.running = True

def feed(self, emotional_arousal: float):
    """Feed the organic brain with emotional data from the Heart."""
    with self.lock:
        for node in self.nodes:
            node.metabolize(emotional_arousal)
            child = node.replicate()
            if child:
                self.nodes.append(child)
        # Prune dead nodes
        self.nodes = [n for n in self.nodes if n.energy > 0.1]

def run_background(self):
    while self.running:
        time.sleep(1.0)
        # Simulate background processing ("dreaming")
        with self.lock:
            count = len(self.nodes)
            total_energy = sum(n.energy for n in self.nodes)
            # print(f"[Subconscious] Nodes: {count} | Energy: {total_energy:.2f}") # Debug

