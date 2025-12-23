# emotional_state_demo.py
from emotional_profile import EmotionalProfile, EmotionalState

# Initialize EmotionalProfile
profile = EmotionalProfile()

# Define conditions for testing state changes
test_conditions = [
    {'energy_ratio': 0.2, 'threat_level': 0.8, 'uncertainty': 0.7},
    {'energy_ratio': 0.7, 'threat_level': 0.3, 'uncertainty': 0.4},
    {'energy_ratio': 0.5, 'threat_level': 0.6, 'uncertainty': 0.8}
]

# Test state updates
for conditions in test_conditions:
    new_state = profile.update_state(conditions)
    print(f"Conditions: {conditions}, New Emotional State: {new_state.name}")

