current_state: EmotionalState = EmotionalState.NEUTRAL
state_intensity: float = 0.5
state_duration: float = 0.0
state_history: List[Tuple[EmotionalState, float, float]] = []

def update_state(self, conditions: Dict) -> EmotionalState:
    state_probs = {
        EmotionalState.ALERT: conditions.get('low_energy', 0) > 0.7,
        EmotionalState.CURIOUS: conditions.get('new_patterns', 0) > 0.5,
        EmotionalState.FOCUSED: conditions.get('task_complexity', 0) > 0.6,
    }
    new_state = max(state_probs, key=state_probs.get, default=self.current_state)

    if new_state != self.current_state:
        self.state_history.append((self.current_state, self.state_intensity, self.state_duration))
        self.current_state = new_state
        self.state_intensity = state_probs[new_state]
        self.state_duration = 0.0
    else:
        self.state_duration += 1.0

    return self.current_state

