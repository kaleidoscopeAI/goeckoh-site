# resource_management_demo.py
from resource_manager import ResourceManager
from emotional_profile import EmotionalState

# Initialize ResourceManager
resource_manager = ResourceManager()

# Test allocation under different emotional states
energy_available = 50.0
modes = ['survival', 'learning', 'growth', 'teaching']

for mode in modes:
    for state in EmotionalState:
        allocation = resource_manager.allocate_resources(energy_available, mode, state)
        print(f"Mode: {mode}, State: {state.name}, Allocated Energy: {allocation}")

