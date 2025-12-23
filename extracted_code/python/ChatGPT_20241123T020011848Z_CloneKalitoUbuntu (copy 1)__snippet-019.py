# self_reflection_demo.py

import numpy as np
from adaptive_ai_node import SelfReflection, EnhancedAdaptiveNode

# Initialize Self Reflection and a test node
self_reflection = SelfReflection()
node = EnhancedAdaptiveNode(node_id="node_01")

# Simulate actions
action_log = []
for i in range(10):  # Shortened for demo
    action_log.append({
        'mode': 'learning',
        'success': np.random.choice([True, False], p=[0.7, 0.3]),
        'energy_used': np.random.uniform(0.1, 1.0),
        'completion_time': np.random.uniform(0.5, 1.5)
    })

# Perform self-reflection based on the action log
insights = self_reflection.reflect(action_log, node.get_state())
print("Self-Reflection Insights:", insights)

