def __init__(self, focus_area):
    self.focus_area = focus_area
    self.members = []

def add_member(self, node):
    self.members.append(node)

def share_knowledge(self, topic, data):
    for member in self.members:
        member.receive_data("community", topic, data)

