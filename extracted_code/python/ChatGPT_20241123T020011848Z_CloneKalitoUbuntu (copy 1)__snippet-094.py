from collections import defaultdict

class MessagingBus:
    def __init__(self):
        self.topics = defaultdict(list)

    def publish(self, topic, data):
        self.topics[topic].append(data)

    def consume(self):
        for topic, messages in self.topics.items():
            if messages:
                return topic, messages.pop(0)
        return None, None

