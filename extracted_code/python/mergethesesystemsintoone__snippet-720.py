"""
Handles communication between nodes, task delegation, and error feedback.
"""

def __init__(self, node_id: str):
    self.node_id = node_id
    self.message_queue = queue.Queue()
    self.active_nodes: Dict[str, Callable] = {}  # Node registry: node_id -> callback function

def register_node(self, node_id: str, callback: Callable):
    """
    Registers a node's callback function for communication.
    :param node_id: Unique ID of the node.
    :param callback: Function to handle incoming messages.
    """
    with threading.Lock():
        if node_id not in self.active_nodes:
            self.active_nodes[node_id] = callback
            logging.info(f"Node {node_id} registered for communication.")
        else:
            logging.warning(f"Node {node_id} is already registered.")

def unregister_node(self, node_id: str):
    """
    Unregisters a node from communication.
    :param node_id: Unique ID of the node to unregister.
    """
    with threading.Lock():
        if node_id in self.active_nodes:
            del self.active_nodes[node_id]
            logging.info(f"Node {node_id} unregistered from communication.")
        else:
            logging.warning(f"Node {node_id} is not registered.")

def send_message(self, target_node_id: str, message: Dict[str, Any]):
    """
    Sends a message to a specific node.
    :param target_node_id: The ID of the recipient node.
    :param message: The message payload.
    """
    if target_node_id in self.active_nodes:
        try:
            self.active_nodes[target_node_id](message)  # Call the target node's callback
            logging.info(f"Message sent to {target_node_id}: {message}")
        except Exception as e:
            logging.error(f"Error sending message to {target_node_id}: {e}")
    else:
        logging.warning(f"Target node {target_node_id} not found.")

def broadcast_message(self, message: Dict[str, Any]):
    """
    Broadcasts a message to all active nodes.
    :param message: The message payload.
    """
    for node_id, callback in self.active_nodes.items():
        try:
            callback(message)
            logging.info(f"Message broadcasted to {node_id}: {message}")
        except Exception as e:
            logging.error(f"Error broadcasting message to {node_id}: {e}")

def receive_message(self, message: Dict[str, Any]):
    """
    Receives a message and adds it to the processing queue.
    :param message: The incoming message payload.
    """
    self.message_queue.put(message)
    logging.info(f"Message received: {message}")

def process_messages(self):
    """
    Processes all messages in the queue.
    """
    while not self.message_queue.empty():
        message = self.message_queue.get()
        logging.info(f"Processing message: {message}")
        # Implement message processing logic here


