def __init__(self, node_id, energy):
    self.node_id = node_id
    self.energy = energy
    self.neighbors = []

def share_resources(self):
    """Share energy with neighbors."""
    if self.energy > 0.5:
        shared_energy = self.energy * 0.1  # Share 10% of energy
        for neighbor in self.neighbors:
            neighbor.energy += shared_energy
        self.energy -= shared_energy * len(self.neighbors)

def connect(self, other_node):
    self.neighbors.append(other_node)
    other_node.neighbors.append(self)

