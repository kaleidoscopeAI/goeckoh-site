class AdaptiveThresholds:
    """Dynamic thresholds that adjust based on performance"""
    base_thresholds: Dict[str, float] = field(default_factory=lambda: {
        'survival': 0.3,
        'growth': 0.5,
        'replication': 0.7,
        'teaching': 0.8
    })
    adaptation_rate: float = 0.05
    history: Dict[str, List[float]] = field(default_factory=lambda: {
        'survival': [],
        'growth': [],
        'replication': [],
        'teaching': []
    })

    def adapt_thresholds(self, performance_history: List[float]):
        """Adjust thresholds based on performance"""
        if not performance_history:
            return

        recent_performance = np.mean(performance_history[-10:])
        for mode, threshold in self.base_thresholds.items():
            if recent_performance > threshold + 0.1:
                # Increase threshold if consistently exceeding it
                self.base_thresholds[mode] += self.adaptation_rate
            elif recent_performance < threshold - 0.1:
                # Decrease threshold if consistently falling short
                self.base_thresholds[mode] = max(
                    0.1,
                    self.base_thresholds[mode] - self.adaptation_rate
                )
            
            self.history[mode].append(self.base_thresholds[mode])

class ContextualPattern:
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

class SharedKnowledgePool:
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

class EnhancedAdaptiveNode:
    """Node with enhanced adaptive capabilities"""
    def __init__(self, node_id: str, shared_pool: SharedKnowledgePool):
        self.id = node_id
        self.shared_pool = shared_pool
        self.weights = DynamicWeights()
        self.thresholds = AdaptiveThresholds()
        self.energy = 10.0
        self.max_energy = 100.0
        self.connections = set()
        self.knowledge_base = {}
        
        # Performance monitoring
        self.performance_history = deque(maxlen=100)
        self.action_outcomes = deque(maxlen=100)
        self.feedback_metrics = defaultdict(list)
        
    def process_with_context(self, data: Dict, context: Dict) -> Dict:
        """Process input with contextual awareness"""
        # Update weights based on current conditions
        self.weights.adapt_to_conditions(
            environment_state=context,
            node_state=self.get_state()
        )
        
        # Create contextual pattern
        pattern = ContextualPattern(data, context)
        
        # Get relevant patterns from shared pool
        relevant_patterns = self.shared_pool.get_relevant_patterns(context)
        
        # Update pattern confidence
        pattern.update_confidence(relevant_patterns)
        
        # Process based on current mode
        score = self.calculate_adaptive_score()
        result = self._process_in_mode(pattern, score)
        
        # Update performance history
        self._update_performance(result)
        
        # Contribute to shared pool if valuable
        if pattern.confidence > 0.7:
            self.shared_pool.contribute_pattern(pattern, self.id)
            
        return result

    def calculate_adaptive_score(self) -> float:
        """Calculate score with dynamic weights"""
        state = self.get_state()
        
        growth_score = len(self.connections) / 10
        efficiency_score = 1 - (self.energy / self.max_energy)
        knowledge_score = len(self.knowledge_base) / 100
        
        score = (
            self.weights.growth * growth_score +
            self.weights.efficiency * efficiency_score +
            self.weights.knowledge * knowledge_score
        )
        
        self.performance_history.append(score)
        return score

    def _process_in_mode(self, pattern: ContextualPattern, score: float) -> Dict:
        """Process pattern based on current mode"""
        # Get adaptive thresholds
        self.thresholds.adapt_thresholds(list(self.performance_history))
        
        # Choose processing mode
        if score < self.thresholds.base_thresholds['survival']:
            return self._survival_mode(pattern)
        elif score < self.thresholds.base_thresholds['growth']:
            return self._learning_mode(pattern)
        elif score < self.thresholds.base_thresholds['teaching']:
            return self._growth_mode(pattern)
        else:
            return self._teaching_mode(pattern)

    def _update_performance(self, result: Dict):
        """Update performance metrics and feedback"""
        # Record action outcome
        self.action_outcomes.append(result)
        
        # Update feedback metrics
        self.feedback_metrics[result['mode']].append({
            'score': self.performance_history[-1],
            'energy_used': result.get('energy_used', 0),
            'effectiveness': result.get('effectiveness', 0),
            'timestamp': time.time()
        })
        
        # Analyze recent performance
        self._analyze_performance()

    def _analyze_performance(self):
        """Analyze performance and adjust behavior"""
        if len(self.action_outcomes) < 10:
            return
            
        recent_outcomes = list(self.action_outcomes)[-10:]
        
        # Calculate success rates per mode
        mode_success = defaultdict(list)
        for outcome in recent_outcomes:
            mode = outcome['mode']
            success = outcome.get('effectiveness', 0) > 0.5
            mode_success[mode].append(success)
            
        # Adjust thresholds based on success rates
        for mode, successes in mode_success.items():
            success_rate = sum(successes) / len(successes)
            if success_rate < 0.3:
                # Lower threshold if struggling
                self.thresholds.base_thresholds[mode] = max(
                    0.1,
                    self.thresholds.base_thresholds[mode] - 0.05
                )
            elif success_rate > 0.8:
                # Raise threshold if too easy
                self.thresholds.base_thresholds[mode] += 0.05

    def get_state(self) -> Dict:
        """Get current node state"""
        return {
            'energy': self.energy,
            'max_energy': self.max_energy,
            'connections': self.connections,
            'knowledge_base': self.knowledge_base
        }

