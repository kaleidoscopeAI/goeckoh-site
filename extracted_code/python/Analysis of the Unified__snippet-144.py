"""Emotional signature extracted from content"""
valence: float = 0.0  # -1 to 1
arousal: float = 0.0  # 0 to 1
coherence: float = 0.5  # 0 to 1
semantic_temperature: float = 1.0

def hyperbolic_distance(self, other: 'EmotionalSignature') -> float:
    """Compute hyperbolic distance in emotional space"""
    # Poincaré disk model distance
    u = np.array([self.valence, self.arousal])
    v = np.array([other.valence, other.arousal])

    norm_u = np.linalg.norm(u)
    norm_v = np.linalg.norm(v)

    # Avoid division by zero
    if norm_u >= 1.0 or norm_v >= 1.0:
        return float('inf')

    delta = np.linalg.norm(u - v)**2
    denom = (1 - norm_u**2) * (1 - norm_v**2)

    if denom <= 0:
        return float('inf')

    return np.arccosh(1 + 2 * delta / denom)

