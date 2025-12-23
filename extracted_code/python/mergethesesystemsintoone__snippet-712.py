def __init__(self):
    """
    Manages the lifecycle of nodes, including creation, replication, and removal.
    """
    self.nodes: Dict[str, Node] = {}

def create_node(self, node_id: Optional[str] = None, dna: Optional[GeneticCode] = None, parent_id: Optional[str] = None) -> str:
    """
    Creates a new node with the specified attributes.

    Args:
        node_id (str): Unique identifier for the node.
        dna (GeneticCode): DNA for the node.
        parent_id (str): ID of the parent node, if it's a replication.

    Returns:
        str: The ID of the newly created node.
    """
    if node_id is None:
        node_id = str(uuid.uuid4())

    if node_id in self.nodes:
        logging.warning(f"Node with ID {node_id} already exists.")
        return None

    new_node = Node(node_id, dna, parent_id)
    self.nodes[node_id] = new_node
    logging.info(f"Node {node_id} created.")
    return node_id

def replicate_node(self, node_id: str) -> Optional[str]:
    """
    Replicates an existing node, with a chance of mutation.

    Args:
        node_id (str): Unique identifier for the node to be replicated.

    Returns:
        Optional[str]: The ID of the new node, or None if replication failed.
    """
    if node_id not in self.nodes:
        logging.warning(f"Node with ID {node_id} does not exist.")
        return None

    parent_node = self.nodes[node_id]
    if parent_node.should_replicate():
        new_node = parent_node.replicate()
        if new_node:
            new_node_id = new_node.node_id
            self.nodes[new_node_id] = new_node
            logging.info(f"Node {node_id} replicated to create node {new_node_id}.")
            return new_node_id
        else:
            logging.info(f"Node {node_id} replication conditions not fully met.")




