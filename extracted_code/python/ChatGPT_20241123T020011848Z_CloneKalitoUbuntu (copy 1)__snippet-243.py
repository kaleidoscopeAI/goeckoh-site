def __init__(self, id, function):
    self.id = id
    self.function = function  # Stores the purpose of the node
    self.connections = []  # Tracks connections to other nodes

def replicate(self):
    new_node = Node(self.id + 1, self.function)  # Simple replication
    self.connections.append(new_node)
    return new_node

def connect(self, other_node):
    if other_node not in self.connections:
        self.connections.append(other_node)

