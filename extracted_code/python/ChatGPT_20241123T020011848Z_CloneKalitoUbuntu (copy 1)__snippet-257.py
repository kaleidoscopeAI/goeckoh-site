# Node learns from experience (simulate experience value with cycle number)
root_node.learn(experience=cycle, impact=cycle % 2 + 1)

# Node replicates to create a child node
child_node = root_node.replicate()

# Environment interaction
resources_received = environment.provide_resources(root_node)
environment.adjust_environment("success" if resources_received > 0 else "low")

# Output state for observation
print(f"Cycle {cycle + 1}: Node {root_node.node_id} traits: {root_node.dna.traits}")
print(f"Resources received: {resources_received}, Environment resources remaining: {environment.resources}")

