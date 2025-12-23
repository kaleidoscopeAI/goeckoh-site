# core_demo.py

from datetime import datetime
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from adaptive_ai_node import EnhancedAdaptiveNode, EmotionalState, ContextualPattern

# Initialize the Enhanced Adaptive Node
node = EnhancedAdaptiveNode(node_id="node_01")

# Sample environment and context conditions
context_conditions = {
    'energy_ratio': 0.7,
    'threat_level': 0.2,
    'uncertainty': 0.3,
    'mode': 'learning'
}

# Sample data inputs for processing
sample_data_1 = {"type": "numeric", "value": np.random.randint(0, 100)}
sample_data_2 = {"type": "text", "content": "Exploring adaptive AI capabilities."}

# Run the node's process method under different emotional states
emotional_states = [
    EmotionalState.ALERT, EmotionalState.CURIOUS, EmotionalState.FOCUSED,
    EmotionalState.SOCIAL, EmotionalState.CONSERVATIVE
]

# Process inputs and capture results
results = []
for state in emotional_states:
    print(f"Processing under {state.name} state.")
    context_conditions['emotional_state'] = state
    result = node.process_input(sample_data_1, context_conditions)
    results.append(result)

# Visualize Emotional Response and Action Log
print("\nEmotional Response & Action Log:")
for result in results:
    print(result)

