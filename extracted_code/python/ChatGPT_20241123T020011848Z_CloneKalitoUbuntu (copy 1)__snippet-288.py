"""Emotional profile affecting decision-making"""
current_state: EmotionalState = EmotionalState.NEUTRAL
state_intensity: float = 0.5
state_duration: float = 0.0
state_history: List[Tuple[EmotionalState, float, float]] = field(default_factory=list)

def update_state(self, conditions: Dict) -> EmotionalState:
    """Update emotional state based on conditions"""
    state_probs = {
        EmotionalState.ALERT: self._calculate_alert_probability(conditions),
        EmotionalState.CURIOUS: self._calculate_curiosity_probability(conditions),
        EmotionalState.FOCUSED: self._calculate_focus_probability(conditions),
        EmotionalState.SOCIAL: self._calculate_social_probability(conditions),
        EmotionalState.CONSERVATIVE: self._calculate_conservative_probability(conditions)
    }

    new_state = max(state_probs.items(), key=lambda x: x[1])

    if new_state[0] != self.current_state:
        self.state_history.append((
            self.current_state,
            self.state_intensity,
            self.state_duration
        ))
        self.current_state = new_state[0]
        self.state_intensity = new_state[1]
        self.state_duration = 0.0
    else:
        self.state_duration += 1.0

    return self.current_state

def _calculate_alert_probability(self, conditions: Dict) -> float:
    alert_factors = [
        conditions.get('energy_ratio', 1.0) < 0.3,
        conditions.get('threat_level', 0.0) > 0.7,
        conditions.get('uncertainty', 0.0) > 0.8
    ]
    return sum(float(f) for f in alert_factors) / len(alert_factors)

