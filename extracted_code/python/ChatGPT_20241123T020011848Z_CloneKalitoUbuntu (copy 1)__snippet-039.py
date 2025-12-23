# complete_demo.py

from enhanced_adaptive_node import EnhancedAdaptiveNode

# Initialize node
node = EnhancedAdaptiveNode(id="test_node")

# Mock input data and context
data = {'pattern': 'Example Data'}
context = {'threat_level': 0.3, 'uncertainty': 0.2}

# Run integrated tests
print("Running full node demo...")

for i in range(5):
    print(f"\n-- Cycle {i+1} --")
    result = node.process_input(data, context)
    print("Processing Result:", result)
    print("Current State:", node.get_state())

