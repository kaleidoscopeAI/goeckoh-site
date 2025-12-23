"""Unified node with adaptive, reflective, and emotional intelligence."""
id: str
emotional_profile: EmotionalProfile = field(default_factory=EmotionalProfile)
self_reflection: SelfReflection = field(default_factory=SelfReflection)
resource_manager: ResourceManager = field(default_factory=ResourceManager)
confidence_calculator: MultiFactorConfidence = field(default_factory=MultiFactorConfidence)
shared_pool: EnhancedSharedKnowledgePool = field(default_factory=EnhancedSharedKnowledgePool)
energy: float = 100.0
max_energy: float = 100.0
action_log: List[Dict] = field(default_factory=list)

def get_state(self) -> Dict:
    """Provide current state for analysis."""
    return {
        'energy': self.energy,
        'emotional_state': self.emotional_profile.current_state.name,
        'knowledge_base': len(self.shared_pool.patterns)
    }

def process_input(self, data: Dict, context: Dict) -> Dict:
    """Integrated processing for all components."""
    # Update emotional state
    self.emotional_profile.update_state(context)

    # Allocate resources based on current state
    allocated_energy = self.resource_manager.allocate_resources(self.energy, 'active', self.emotional_profile.current_state)

    # Calculate confidence of data
    confidence = self.confidence_calculator.calculate_confidence(data, context)

    # Store action result
    result = {
        'mode': 'active',
        'energy_used': allocated_energy,
        'confidence': confidence,
        'emotional_state': self.emotional_profile.current_state.name
    }
    self.action_log.append(result)

    # Periodic reflection and knowledge update
    if len(self.action_log) >= 5:
        insights = self.self_reflection.reflect(self.action_log, self.get_state())
        self.shared_pool.add_insight(insights)
        self.action_log.clear()

    return result

