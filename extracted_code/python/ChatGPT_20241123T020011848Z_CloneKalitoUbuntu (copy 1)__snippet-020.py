# adaptive_ai_node.py

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import numpy as np
import time
import enum

class EmotionalState(enum.Enum):
    NEUTRAL = "neutral"
    ALERT = "alert"
    CURIOUS = "curious"
    FOCUSED = "focused"
    SOCIAL = "social"
    CONSERVATIVE = "conservative"

@dataclass
class EmotionalProfile:
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

class EnhancedAdaptiveNode:
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

