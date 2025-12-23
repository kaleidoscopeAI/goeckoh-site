def __init__(self):
    self.nodes = {}

def get_element_by_id(self, node_id):
    return self.nodes.get(node_id, "Node not found")

def add_node(self, node_id, node_data):
    self.nodes[node_id] = node_data
