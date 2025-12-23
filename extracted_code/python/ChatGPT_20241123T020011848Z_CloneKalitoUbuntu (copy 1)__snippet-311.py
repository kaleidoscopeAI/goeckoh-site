def __init__(self, nodes):
    self.nodes = nodes

def broadcast(self, message, sender):
    """Broadcast a message to all nodes."""
    for node in self.nodes:
        if node != sender:
            node.learn(message)

def unicast(self, message, sender, receiver):
    """Send a message to a specific node."""
    receiver.learn(message)

def sync_knowledge(self, node_a, node_b):
    """Synchronize knowledge between two nodes."""
    combined_knowledge = {**node_a.knowledge, **node_b.knowledge}
    node_a.knowledge = combined_knowledge
    node_b.knowledge = combined_knowledge

