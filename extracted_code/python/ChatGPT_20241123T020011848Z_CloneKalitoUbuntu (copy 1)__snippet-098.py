from core_node import Node
import numpy as np

def create_seed_node():
    """Create the first seed node."""
    dna = np.random.rand(5)  # Initialize DNA with random values
    seed_node = Node(node_id=1, dna=dna)
    return seed_node

def replicate_node(node, new_node_id):
    """Replicate an existing node."""
    return node.replicate(new_node_id)

