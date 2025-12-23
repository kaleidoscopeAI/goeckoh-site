def __init__(self, node_id: str):
    self.id = node_id
    self.emotional_profile = EmotionalProfile()
    self.energy = 10.0
    self.action_history = []

def process_input(self, data: Dict) -> Dict:
    emotional_state = self.emotional_profile.update_state(data)
    result = {"id": self.id, "state": str(emotional_state), "processed_data": data}
    self.action_history.append(result)
    return result

