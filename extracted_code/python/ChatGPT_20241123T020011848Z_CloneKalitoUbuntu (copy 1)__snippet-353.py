def __init__(self):
    self.index = {}

def add_entry(self, topic, reference):
    if topic not in self.index:
        self.index[topic] = []
    if reference not in self.index[topic]:
        self.index[topic].append(reference)

def get_references(self, topic):
    return self.index.get(topic, [])

