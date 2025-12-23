from core import Node
from node_creation import create_new_node
from learning import crawl_and_learn
from resource_management import manage_resources
from mirrored_network import MirroredNetwork

# Initialize nodes and mirrored network
initial_node = Node(0, "seed_dna", {"energy": 1.0})
nodes = [initial_node]
network = MirroredNetwork()
network.add_node(initial_node)

# Run simulation
threshold = 2
topics = ["Artificial Intelligence", "Hot Dog", "Elephant"]

for epoch in range(10):
    print(f"Epoch {epoch}:")

    # Learn and share knowledge
    for node in nodes:
        topic = topics[epoch % len(topics)]
        crawl_and_learn(node, topic)
    
    # Manage resources
    manage_resources(nodes)

    # Replicate nodes if threshold met
    new_node = create_new_node(nodes, "seed_dna", threshold)
    if new_node:
        network.add_node(new_node)
        network.add_edge(new_node.node_id, initial_node.node_id)

    # Visualize network
    if epoch % 5 == 0:
        network.visualize()

# Save logs
for node in nodes:
    with open(f"node_{node.node_id}.json", "w") as log_file:
        log_file.write(node.to_json())

