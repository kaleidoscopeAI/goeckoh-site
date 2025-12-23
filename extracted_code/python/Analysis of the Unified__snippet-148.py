"""
Implements the CIAE equation for intelligent crawling
CIAE(d,t) = α·I_Q + β·E_H + γ·R_E + δ·A_M
"""

def __init__(self, 
             alpha: float = 0.35,  # Quantum information weight
             beta: float = 0.25,   # Emotional distance weight
             gamma: float = 0.30,  # Relational entanglement weight
             delta: float = 0.10): # Adaptive meta-learning weight
    self.alpha = alpha
    self.beta = beta
    self.gamma = gamma
    self.delta = delta

    self.visited_embeddings = deque(maxlen=1000)
    self.system_emotional_state = EmotionalSignature()
    self.knowledge_graph = {}
    self.success_history = deque(maxlen=100)

def compute_ciae_score(self, 
                       doc: CrawledDocument,
                       context_embeddings: List[np.ndarray]) -> float:
    """Compute CIAE score for document prioritization"""

    # Component 1: Quantum Information Gain
    I_Q = self._quantum_information_gain(doc, context_embeddings)

    # Component 2: Emotional Hyperbolic Distance
    E_H = self._emotional_hyperbolic_score(doc)

    # Component 3: Relational Entanglement
    R_E = self._relational_entanglement(doc)

    # Component 4: Adaptive Meta-learning
    A_M = self._adaptive_metalearning()

    # Compute weighted CIAE score
    ciae_score = (self.alpha * I_Q + 
                 self.beta * E_H + 
                 self.gamma * R_E + 
                 self.delta * A_M)

    return float(np.clip(ciae_score, 0, 1))

def _quantum_information_gain(self, 
                              doc: CrawledDocument,
                              context: List[np.ndarray]) -> float:
    """Quantum-inspired information gain"""
    if not context or doc.embedding is None:
        return 1.0

    # Compute quantum interference with existing knowledge
    similarities = [np.dot(doc.embedding, ctx) for ctx in context]

    # Quantum probability amplitude
    avg_similarity = np.mean(similarities)
    max_similarity = np.max(similarities)

    # Information gain from quantum measurement
    if avg_similarity > 0:
        info_gain = max_similarity * np.log(max_similarity / avg_similarity)
    else:
        info_gain = 1.0

    # Modulate by quantum state measurement
    quantum_prob = doc.quantum_state.measure_probability()

    return float(np.clip(info_gain * quantum_prob, 0, 1))

def _emotional_hyperbolic_score(self, doc: CrawledDocument) -> float:
    """Emotional hyperbolic distance scoring"""
    # Compute hyperbolic distance in emotional space
    distance = doc.emotional_sig.hyperbolic_distance(
        self.system_emotional_state)

    # Convert distance to score (closer = higher score)
    if np.isinf(distance):
        return 0.0

    # Use Gaussian kernel for smooth scoring
    score = np.exp(-distance**2 / 2)

    return float(np.clip(score, 0, 1))

def _relational_entanglement(self, doc: CrawledDocument) -> float:
    """Measure relational entanglement with knowledge graph"""
    if not self.knowledge_graph:
        return 0.5

    # Count semantic overlaps with existing knowledge
    entanglement = 0.0
    doc_words = set(doc.content.lower().split()[:100])

    for kg_entry in list(self.knowledge_graph.values())[-50:]:
        kg_words = set(kg_entry.get('content', '').lower().split()[:100])
        overlap = len(doc_words & kg_words)
        entanglement += overlap / (len(doc_words) + 1)

    return float(np.clip(entanglement / 50, 0, 1))

def _adaptive_metalearning(self) -> float:
    """Adaptive factor based on system learning history"""
    if len(self.success_history) < 10:
        return 0.5

    recent_success = np.mean(list(self.success_history)[-20:])
    return float(np.tanh(2 * recent_success - 1) * 0.5 + 0.5)

def update_system_state(self, doc: CrawledDocument, success: bool):
    """Update system state after document ingestion"""
    if doc.embedding is not None:
        self.visited_embeddings.append(doc.embedding)

    # Update emotional state (running average)
    alpha = 0.1
    self.system_emotional_state.valence = (
        (1 - alpha) * self.system_emotional_state.valence +
        alpha * doc.emotional_sig.valence
    )
    self.system_emotional_state.arousal = (
        (1 - alpha) * self.system_emotional_state.arousal +
        alpha * doc.emotional_sig.arousal
    )

    self.success_history.append(1.0 if success else 0.0)

    # Add to knowledge graph
    self.knowledge_graph[doc.url] = {
        'content': doc.content[:500],
        'timestamp': doc.timestamp,
        'ciae_score': doc.ciae_score
    }

