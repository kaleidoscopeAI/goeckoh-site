def __init__(self):
    self.synaptic_map = {}  # Key: Node ID, Value: Learned knowledge

def add_synapse(self, node_id, knowledge):
    """Link knowledge to a node's synapse."""
    if node_id not in self.synaptic_map:
        self.synaptic_map[node_id] = []
    self.synaptic_map[node_id].append(knowledge)

def query_synapse(self, query):
    """Search across synaptic connections for relevant knowledge."""
    results = {}
    for node_id, knowledge_list in self.synaptic_map.items():
        matches = [k for k in knowledge_list if query in k]
        if matches:
            results[node_id] = matches
    return results

