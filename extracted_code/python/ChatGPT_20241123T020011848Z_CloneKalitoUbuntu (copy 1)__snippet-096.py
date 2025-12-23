from nodes.node_template import SpecializedNode
from visualization.node_growth_visual import visualize_growth

# Initialize nodes
nodes = [SpecializedNode(node_id=1, specialization="object_detection")]

# Simulate environment
for _ in range(50):
    for node in nodes:
        node.process()
        new_node = node.grow()
        if new_node:
            nodes.append(new_node)

# Visualize final growth
visualize_growth(nodes)

