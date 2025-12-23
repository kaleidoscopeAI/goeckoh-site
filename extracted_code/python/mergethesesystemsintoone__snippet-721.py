# ... (previous methods) ...

def share_knowledge(self, other_node_id, knowledge_key):
    """Share a piece of knowledge with another node."""
    if knowledge_key in self.knowledge_base:
        message = {
            "type": "knowledge_sharing",
            "sender": self.node_id,
            "knowledge_key": knowledge_key,
            "knowledge": self.knowledge_base[knowledge_key]
        }
        self.comm.send_message(other_node_id, message)  # Assuming self.comm is the NodeCommunication instance
        self.log_event(f"Node {self.node_id} shared knowledge '{knowledge_key}' with Node {other_node_id}.")
    else:
        self.log_event(f"Node {self.node_id} attempted to share unknown knowledge '{knowledge_key}'.")

def receive_knowledge(self, message: Dict):
    """Receive knowledge from another node."""
    knowledge_key = message.get("knowledge_key")
    knowledge_value = message.get("knowledge")
    if knowledge_key and knowledge_value:
        if knowledge_key not in self.knowledge_base:
            self.knowledge_base[knowledge_key] = []
        self.knowledge_base[knowledge_key].extend(knowledge_value)
        self.log_event(f"Node {self.node_id} received knowledge '{knowledge_key}'.")


