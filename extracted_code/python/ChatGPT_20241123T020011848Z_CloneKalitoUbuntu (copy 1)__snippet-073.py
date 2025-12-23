shared_hub = {}

def share_with_hub(node_id, data):
    shared_hub[node_id] = data

def retrieve_from_hub(node_id):
    return shared_hub.get(node_id, "No data available.")

