"""Distributed knowledge pool for collaborative learning"""
def __init__(self):
    self.patterns: Dict[str, ContextualPattern] = {}
    self.node_contributions: Dict[str, Set[str]] = defaultdict(set)
    self.pattern_relationships = nx.Graph()
    self.access_history = defaultdict(list)

def contribute_pattern(self, pattern: ContextualPattern, node_id: str):
    """Add pattern to shared pool"""
    pattern_id = str(uuid.uuid4())
    self.patterns[pattern_id] = pattern
    self.node_contributions[node_id].add(pattern_id)

    # Update pattern relationships
    self._update_pattern_relationships(pattern_id)

def _update_pattern_relationships(self, new_pattern_id: str):
    """Update relationships between patterns"""
    new_pattern = self.patterns[new_pattern_id]
    self.pattern_relationships.add_node(new_pattern_id)

    for pid, pattern in self.patterns.items():
        if pid != new_pattern_id:
            similarity = new_pattern._calculate_context_similarity(pattern)
            if similarity > 0.5:
                self.pattern_relationships.add_edge(
                    new_pattern_id,
                    pid,
                    weight=similarity
                )

def get_relevant_patterns(self, context: Dict, limit: int = 10) -> List[ContextualPattern]:
    """Get patterns relevant to given context"""
    relevance_scores = {}

    for pid, pattern in self.patterns.items():
        similarity = pattern._calculate_context_similarity(
            ContextualPattern({}, context)
        )
        relevance_scores[pid] = similarity

    # Get most relevant patterns
    relevant_ids = sorted(
        relevance_scores.keys(),
        key=lambda x: relevance_scores[x],
        reverse=True
    )[:limit]

    return [self.patterns[pid] for pid in relevant_ids]

