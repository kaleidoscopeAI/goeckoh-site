def __init__(self):
    self.resource_threshold = 0.5

def allocate_resources(self, nodes):
    """Balance resources across all nodes."""
    total_energy = sum(node.energy for node in nodes)
    average_energy = total_energy / len(nodes)

    for node in nodes:
        if node.energy < average_energy * self.resource_threshold:
            node.energy += 0.1
        else:
            node.energy -= 0.05

def monitor_resources(self, nodes):
    return {
        "Average Energy": sum(node.energy for node in nodes) / len(nodes),
        "Node Count": len(nodes)
    }

