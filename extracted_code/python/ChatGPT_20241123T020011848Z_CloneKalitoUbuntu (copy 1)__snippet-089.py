class Node:
    def __init__(self, node_id, library):
        self.node_id = node_id
        self.library = library
        self.capabilities = []

    def request_capability(self, capability_name):
        resource = self.library.get_resource(capability_name)
        if resource != "Resource not available.":
            self.capabilities.append(capability_name)
            self.logs.append(f"Acquired capability: {capability_name}")
        else:
            self.logs.append(f"Failed to acquire capability: {capability_name}")

