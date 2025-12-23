import uuid
import time
import random

class Node:
    def __init__(self):
        self.id = str(uuid.uuid4())
        self.maturity = 0.01
        self.energy = 1.0
        self.knowledge = {}
        self.connections = set()
        self.generation = 0

    def learn(self, data):
        """Learn from given data and grow maturity."""
        knowledge_gain = len(data) * 0.01
        self.knowledge.update(data)
        self.maturity += knowledge_gain
        self.energy -= knowledge_gain * 0.05

    def share_resources(self, target_node):
        """Share resources like energy or knowledge."""
        energy_transfer = min(0.1, self.energy * 0.2)
        if self.energy > energy_transfer:
            self.energy -= energy_transfer
            target_node.energy += energy_transfer

    def adapt(self):
        """Adapt energy usage based on maturity."""
        self.energy -= 0.01 * self.maturity
        if self.energy < 0.2:
            self.energy += random.uniform(0.1, 0.3)  # Simulate optimization

    def status(self):
        return {
            "ID": self.id,
            "Maturity": self.maturity,
            "Energy": self.energy,
            "Knowledge Size": len(self.knowledge),
            "Connections": len(self.connections)
        }

