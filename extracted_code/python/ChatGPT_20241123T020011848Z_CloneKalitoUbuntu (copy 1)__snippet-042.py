class ResourceManager:
    def allocate(self, nodes):
        """Allocate resources dynamically among nodes."""
        total_energy = sum(node.energy for node in nodes)
        average_energy = total_energy / len(nodes)
        for node in nodes:
            if node.energy < average_energy * 0.8:
                node.energy += 0.1
            else:
                node.energy -= 0.05

