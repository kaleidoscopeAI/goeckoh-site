def __post_init__(self):
    self._validate()
    self._initialize_metadata()

def _validate(self):
    if not 0 <= self.value <= 1:
        raise ValueError(f"Trait value must be between 0 and 1, got {self.value}")
    if not 0 <= self.plasticity <= 1:
        raise ValueError(f"Plasticity must be between 0 and 1, got {self.plasticity}")

def _initialize_metadata(self):
    self.history = []
    self.adaptation_score = 1.0
    self.last_update = 0
    self.interaction_strength = {}
