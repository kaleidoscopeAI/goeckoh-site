# shared_pool_demo.py

import networkx as nx
import matplotlib.pyplot as plt
from adaptive_ai_node import EnhancedSharedKnowledgePool, ContextualPattern

# Initialize shared knowledge pool
shared_pool = EnhancedSharedKnowledgePool()

# Sample patterns with different contexts
patterns = [
    ContextualPattern({"content": "AI model"}, {"context": "learning"}),
    ContextualPattern({"content": "self-reflection"}, {"context": "adaptation"}),
    ContextualPattern({"content": "resource management"}, {"context": "conservation"})
]

# Add patterns to the shared pool and update relationships
for pattern in patterns:
    shared_pool.add_pattern(pattern=pattern.content, confidence=0.8)

# Visualize relationship graph
plt.figure(figsize=(8, 8))
pos = nx.spring_layout(shared_pool.pattern_graph)
nx.draw_networkx(shared_pool.pattern_graph, pos, with_labels=True, node_size=700, node_color='skyblue')
plt.title("Pattern Relationship Graph in Shared Knowledge Pool")
plt.show()

