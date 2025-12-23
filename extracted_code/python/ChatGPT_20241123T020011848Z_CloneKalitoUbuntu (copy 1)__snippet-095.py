from core import Node
from messaging import MessagingBus
from resource_library import ResourceLibrary
from knowledge_index import KnowledgeIndex
from visualization.node_growth_visual import visualize_node_growth

# Initialize environment
messaging_bus = MessagingBus()
resource_library = ResourceLibrary()
knowledge_index = KnowledgeIndex()

# Create initial nodes
nodes = [Node(node_id=1, messaging_bus=messaging_bus, resource_library=resource_library, knowledge_index=knowledge_index)]

# Simulate growth
for epoch in range(20):
    new_nodes = []
    for node in nodes:
        node.process()
        new_node = node.grow()
        if new_node:
            new_nodes.append(new_node)
    nodes.extend(new_nodes)

# Visualize final network
visualize_node_growth(nodes)

