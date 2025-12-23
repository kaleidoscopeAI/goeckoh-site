# Full AI Ecosystem Simulation
environment = Environment()
root_node = EnhancedOrganicCore(node_id="root", dna=EnhancedNodeDNA(traits={
    'learning_capacity': 1.0,
    'adaptation_rate': 1.0,
    'resilience': 1.0,
    'efficiency': 1.0,
    'specialization': 1.0
}))

# Simulate 10 growth cycles
for cycle in range(10):
    # Node learns and adapts
    experience = {'impact': cycle * 0.5}
    root_node.process_experience(experience)
    
    # Node replicates and forms connections with child
    child_node = root_node.replicate()
    root_node.connections.add(child_node.node_id)
    
    # Environment provides resources
    resources_received = environment.provide_resources(root_node)
    
    # Output state for observation
    print(f"Cycle {cycle + 1}: Node {root_node.node_id} traits: {root_node.dna.traits}")
    print(f"Resources received: {resources_received}, Environment resources remaining: {environment.resources}")

    # Adjust environment based on success
    environment.adjust_environment("success" if resources_received > 0 else "low")

