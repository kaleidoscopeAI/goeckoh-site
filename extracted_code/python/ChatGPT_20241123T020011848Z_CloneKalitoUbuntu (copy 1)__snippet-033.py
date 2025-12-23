# self_reflection_demo.py
from reflection_analysis import SelfReflection
from adaptive_ai_node import EnhancedAdaptiveNode

# Initialize SelfReflection and a sample node
self_reflection = SelfReflection()
node = EnhancedAdaptiveNode(node_id="test_node")

# Mock action log to simulate recent actions
action_log = [
    {'mode': 'learning', 'success': True, 'energy_used': 5, 'completion_time': 2.0},
    {'mode': 'growth', 'success': False, 'energy_used': 10, 'completion_time': 4.0},
    {'mode': 'teaching', 'success': True, 'energy_used': 8, 'completion_time': 3.0},
]

# Run reflection
insights = self_reflection.reflect(action_log, node.get_state())
print("Reflection Insights:", insights)

