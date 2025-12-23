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

    logging.info(f"Node {node_id} did not replicate (criteria not met or replication failed).")
    return None

def remove_node(self, node_id: str):
    """
    Removes a node from the system.

    Args:
        node_id (str): Unique identifier for the node to be removed.

    Returns:
        None
    """
    if node_id in self.nodes:
        del self.nodes[node_id]
        logging.info(f"Node {node_id} removed from the system.")
    else:
        logging.warning(f"Attempted to remove non-existent node {node_id}.")

def get_node_status(self, node_id: str) -> Dict:
    """
    Retrieves the status of a specific node.

    Args:
        node_id (str): Unique identifier for the node.

    Returns:
        Dict: Status of the node.
    """
    if node_id in self.nodes:
        return self.nodes[node_id].get_status()
    else:
        logging.warning(f"Node {node_id} not found.")
        return {}

def get_all_nodes_status(self) -> Dict[str, Dict]:
    """
    Retrieves the status of all nodes in the system.

    Returns:
        Dict[str, Dict]: Status of all nodes.
    """
    return {node_id: node.get_status() for node_id, node in self.nodes.items()}

