# Node learns from experience
root_node.learn(cycle)  # Each cycle is a new experience

# Replicate the node
child_node = root_node.replicate()

# Environment provides resources to the node
resources_received = environment.provide_resources(root_node)
print(f"Cycle {cycle}: Node {root_node.node_id} adapted and replicated.")
print(f"Resources received: {resources_received}")

# Environment adjusts based on node's success
environment.adjust_environment("success" if resources_received > 0 else "low")

