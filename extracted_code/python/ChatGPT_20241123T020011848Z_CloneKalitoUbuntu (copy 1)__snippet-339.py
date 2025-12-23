def __init__(self):
    self.resource_pool = {}

def share_resource(self, resource_id, resource_content, originating_node_id):
    """Share a resource across nodes."""
    if resource_id not in self.resource_pool:
        self.resource_pool[resource_id] = {
            "content": resource_content,
            "origin": originating_node_id,
            "usage_count": 0
        }

def query_resource(self, resource_id):
    """Query a specific resource."""
    return self.resource_pool.get(resource_id)

def mark_resource_usage(self, resource_id):
    """Increment usage count for tracking."""
    if resource_id in self.resource_pool:
        self.resource_pool[resource_id]["usage_count"] += 1

