class CommunicationNode(AdaptiveNode):
    def send_message(self, target_node, data):
        if target_node in self.connections:
            target_node.receive_message(data)
    
    def receive_message(self, data):
        self.learn(data)  # Incorporate the received data into the learning process

