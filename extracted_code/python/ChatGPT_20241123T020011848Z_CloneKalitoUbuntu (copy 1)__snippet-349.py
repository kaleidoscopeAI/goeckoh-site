def __init__(self, node_id):
    self.node_id = node_id
    self.knowledge = {}
    self.subscriptions = []
    self.logs = []

def subscribe_to_topic(self, topic):
    self.subscriptions.append(topic)
    self.logs.append(f"Subscribed to topic: {topic}")

def publish(self, topic, data):
    # Broadcast data to nodes subscribed to the topic
    for node in nodes:
        if topic in node.subscriptions:
            node.receive_data(self.node_id, topic, data)

def receive_data(self, sender_id, topic, data):
    self.logs.append(f"Received data from Node {sender_id} on topic '{topic}': {data}")
    self.knowledge[topic] = data

