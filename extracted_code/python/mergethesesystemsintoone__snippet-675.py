# ... existing methods ...

def distribute_data_to_nodes(self, nodes: List[Node], data: Any):
    """
    Distributes data among nodes based on their specialization or other criteria.

    Args:
        nodes: List of Node instances.
        data: Data to be distributed.
    """
    for node in nodes:
        # Basic distribution logic (can be made more sophisticated)
        if node.state.status == "Active":
            node.process_data(data)
            logging.info(f"Data distributed to node {node.node_id}")

# ... other methods ...



