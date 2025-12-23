def __init__(self):
    self.primary_node = Node()
    self.mirrored_node = Node()

def cross_validate(self):
    """Cross-validate knowledge between nodes."""
    common_keys = set(self.primary_node.knowledge.keys()) & set(self.mirrored_node.knowledge.keys())
    unique_primary = set(self.primary_node.knowledge.keys()) - common_keys
    unique_mirrored = set(self.mirrored_node.knowledge.keys()) - common_keys

    return {
        "Common Knowledge": len(common_keys),
        "Primary Unique": len(unique_primary),
        "Mirrored Unique": len(unique_mirrored)
    }

def simulate(self, iterations=50):
    """Simulate the learning and mirroring process."""
    for i in range(iterations):
        self.primary_node.learn({"Topic": f"Data-{i}"})
        self.mirrored_node.learn({"Topic": f"Data-{i}"})
        self.primary_node.share_resources(self.mirrored_node)
        self.mirrored_node.share_resources(self.primary_node)
        print(self.cross_validate())

