"""Pattern with contextual awareness"""
def __init__(self, content: Dict, context: Dict):
    self.content = content
    self.context = context
    self.timestamp = time.time()
    self.confidence = 0.5
    self.related_patterns = set()
    self.impact_score = 0.0

def update_confidence(self, similar_patterns: List['ContextualPattern']):
    """Update confidence based on similar patterns"""
    if not similar_patterns:
        return

    confidence_sum = sum(p.confidence for p in similar_patterns)
    context_similarity = np.mean([
        self._calculate_context_similarity(p) for p in similar_patterns
    ])

    self.confidence = (
        0.7 * self.confidence +
        0.3 * (confidence_sum / len(similar_patterns)) * context_similarity
    )

def _calculate_context_similarity(self, other: 'ContextualPattern') -> float:
    """Calculate similarity between contexts"""
    if not self.context or not other.context:
        return 0.0

    common_keys = set(self.context.keys()) & set(other.context.keys())
    if not common_keys:
        return 0.0

    similarities = []
    for key in common_keys:
        if isinstance(self.context[key], (int, float)) and \
           isinstance(other.context[key], (int, float)):
            # Numerical comparison
            max_val = max(abs(self.context[key]), abs(other.context[key]))
            if max_val == 0:
                similarities.append(1.0)
            else:
                similarities.append(
                    1 - abs(self.context[key] - other.context[key]) / max_val
                )
        else:
            # String comparison
            similarities.append(
                1.0 if self.context[key] == other.context[key] else 0.0
            )

    return np.mean(similarities)

