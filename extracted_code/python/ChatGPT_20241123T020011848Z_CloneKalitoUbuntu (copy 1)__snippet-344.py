def __init__(self, node_id, dna, resources):
    self.node_id = node_id
    self.dna = dna
    self.resources = resources
    self.logs = []

def replicate(self, threshold, node_id_counter):
    if len(self.resources) >= threshold:
        child_dna = self.dna + f"_child{node_id_counter}"
        return Node(node_id_counter, child_dna, resources={})
    return None

def communicate(self, other_node):
    shared_data = {"shared_resources": list(self.resources.keys())}
    other_node.receive_data(shared_data)

def receive_data(self, data):
    self.logs.append(f"Received data: {data}")

def to_json(self):
    return json.dumps({
        "node_id": self.node_id,
        "dna": self.dna,
        "resources": self.resources,
        "logs": self.logs
    })

