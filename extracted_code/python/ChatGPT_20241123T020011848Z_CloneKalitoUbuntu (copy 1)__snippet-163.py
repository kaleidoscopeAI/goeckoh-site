class ObjectiveMetrics:
    """Metrics used in objective score calculation"""
    growth_factor: float = 0.0
    energy_usage: float = 0.0
    knowledge_gain: float = 0.0
    total_score: float = 0.0
    timestamp: float = 0.0
    
    def __post_init__(self):
        self.timestamp = time.time()

class ObjectiveDrivenNode:
    """Node with clear objective scoring system"""
    def __init__(self):
        # Core attributes
        self.energy = 10.0
        self.max_energy = 100.0
        self.knowledge_base = {}
        self.connections = set()
        self.maturity = 0.0
        
        # Objective system
        self.weights = ObjectiveWeights()
        self.metrics_history = []
        self.current_metrics = ObjectiveMetrics()
        
        # Performance thresholds
        self.score_thresholds = {
            'survival': 0.3,     # Minimum score for basic operation
            'growth': 0.5,       # Score needed to grow
            'replication': 0.7,  # Score needed to replicate
            'teaching': 0.8      # Score needed to teach other nodes
        }
        
        # Initialize tracking
        self._update_metrics()

    def calculate_objective_score(self) -> float:
        """Calculate overall objective score"""
        # Calculate growth factor (0 to 1)
        growth_factor = min(1.0, len(self.connections) / 10 + self.maturity)
        
        # Calculate energy efficiency (0 to 1)
        energy_efficiency = 1.0 - (self.energy / self.max_energy)
        
        # Calculate knowledge factor (0 to 1)
        knowledge_factor = min(1.0, len(self.knowledge_base) / 100)
        
        # Calculate weighted score
        score = (
            self.weights.growth * growth_factor +
            self.weights.efficiency * energy_efficiency +
            self.weights.knowledge * knowledge_factor
        )
        
        # Update metrics
        self.current_metrics = ObjectiveMetrics(
            growth_factor=growth_factor,
            energy_usage=self.energy,
            knowledge_gain=knowledge_factor,
            total_score=score
        )
        self.metrics_history.append(self.current_metrics)
        
        return score

    def process_input(self, data: Dict) -> Dict:
        """Process input based on current objective score"""
        score = self.calculate_objective_score()
        
        # Determine action based on score
        if score < self.score_thresholds['survival']:
            return self._survival_mode(data)
        elif score < self.score_thresholds['growth']:
            return self._learning_mode(data)
        elif score < self.score_thresholds['replication']:
            return self._growth_mode(data)
        else:
            return self._advanced_mode(data)

    def _survival_mode(self, data: Dict) -> Dict:
        """Operate in survival mode - minimize energy usage"""
        try:
            # Process only essential patterns
            essential_patterns = self._extract_essential_patterns(data)
            
            # Minimal energy usage
            energy_cost = len(essential_patterns) * 0.1
            if self.energy >= energy_cost:
                self.energy -= energy_cost
                self._learn_patterns(essential_patterns, minimal=True)
                
            return {
                'mode': 'survival',
                'patterns_processed': len(essential_patterns),
                'energy_used': energy_cost
            }
            
        except Exception as e:
            return {'error': str(e), 'mode': 'survival_failed'}

    def _learning_mode(self, data: Dict) -> Dict:
        """Operate in learning mode - focus on knowledge acquisition"""
        try:
            # Extract and process patterns
            patterns = self._extract_patterns(data)
            
            # Moderate energy usage
            energy_cost = len(patterns) * 0.2
            if self.energy >= energy_cost:
                self.energy -= energy_cost
                knowledge_gained = self._learn_patterns(patterns)
                
                return {
                    'mode': 'learning',
                    'patterns_processed': len(patterns),
                    'knowledge_gained': knowledge_gained,
                    'energy_used': energy_cost
                }
            
            return {'mode': 'learning_insufficient_energy'}
            
        except Exception as e:
            return {'error': str(e), 'mode': 'learning_failed'}

    def _growth_mode(self, data: Dict) -> Dict:
        """Operate in growth mode - form connections and mature"""
        try:
            # Process patterns and form connections
            patterns = self._extract_patterns(data)
            connections_formed = self._form_connections(patterns)
            
            # Higher energy usage for growth
            energy_cost = (len(patterns) * 0.2 + len(connections_formed) * 0.3)
            if self.energy >= energy_cost:
                self.energy -= energy_cost
                self.maturity = min(1.0, self.maturity + 0.1)
                
                return {
                    'mode': 'growth',
                    'patterns_processed': len(patterns),
                    'connections_formed': len(connections_formed),
                    'maturity_level': self.maturity,
                    'energy_used': energy_cost
                }
            
            return {'mode': 'growth_insufficient_energy'}
            
        except Exception as e:
            return {'error': str(e), 'mode': 'growth_failed'}

    def _advanced_mode(self, data: Dict) -> Dict:
        """Operate in advanced mode - teach and optimize"""
        try:
            # Process patterns and optimize knowledge
            patterns = self._extract_patterns(data)
            optimizations = self._optimize_knowledge(patterns)
            teachings = self._teach_connected_nodes(patterns)
            
            # High energy usage for advanced operations
            energy_cost = (
                len(patterns) * 0.2 +
                len(optimizations) * 0.3 +
                len(teachings) * 0.4
            )
            
            if self.energy >= energy_cost:
                self.energy -= energy_cost
                
                return {
                    'mode': 'advanced',
                    'patterns_processed': len(patterns),
                    'optimizations': len(optimizations),
                    'teachings': len(teachings),
                    'energy_used': energy_cost
                }
            
            return {'mode': 'advanced_insufficient_energy'}
            
        except Exception as e:
            return {'error': str(e), 'mode': 'advanced_failed'}

    def _extract_essential_patterns(self, data: Dict) -> List[Dict]:
        """Extract only the most essential patterns to save energy"""
        patterns = []
        try:
            if isinstance(data, dict):
                for key, value in data.items():
                    if self._is_essential_pattern(key, value):
                        patterns.append({
                            'type': 'essential',
                            'key': key,
                            'value': value,
                            'priority': 1.0
                        })
            return patterns[:3]  # Limit to top 3 essential patterns
        except Exception:
            return patterns

    def _is_essential_pattern(self, key: str, value: any) -> bool:
        """Determine if a pattern is essential for survival"""
        try:
            # Check if pattern relates to energy or critical knowledge
            if 'energy' in str(key).lower():
                return True
            if any(critical in str(value).lower() 
                  for critical in ['survival', 'essential', 'critical']):
                return True
            return False
        except Exception:
            return False

    def _learn_patterns(self, patterns: List[Dict], minimal: bool = False) -> float:
        """Learn from patterns with optional minimal processing"""
        knowledge_gained = 0.0
        try:
            for pattern in patterns:
                if minimal:
                    # Minimal processing - store only essential info
                    if pattern.get('type') == 'essential':
                        self.knowledge_base[pattern['key']] = {
                            'value': pattern['value'],
                            'timestamp': time.time()
                        }
                        knowledge_gained += 0.1
                else:
                    # Full processing
                    self.knowledge_base[pattern['key']] = {
                        'value': pattern['value'],
                        'timestamp': time.time(),
                        'connections': [],
                        'confidence': 0.5
                    }
                    knowledge_gained += 0.2
                    
            return knowledge_gained
        except Exception:
            return knowledge_gained

    def _optimize_knowledge(self, patterns: List[Dict]) -> List[Dict]:
        """Optimize existing knowledge based on new patterns"""
        optimizations = []
        try:
            for pattern in patterns:
                for key, knowledge in self.knowledge_base.items():
                    if self._are_patterns_related(pattern, knowledge):
                        # Create optimization record
                        optimization = {
                            'source': pattern,
                            'target': key,
                            'type': 'enhancement',
                            'timestamp': time.time()
                        }
                        optimizations.append(optimization)
                        
                        # Update knowledge
                        self.knowledge_base[key]['confidence'] = min(
                            1.0,
                            knowledge['confidence'] + 0.1
                        )
            return optimizations
        except Exception:
            return optimizations

    def _teach_connected_nodes(self, patterns: List[Dict]) -> List[Dict]:
        """Share knowledge with connected nodes"""
        teachings = []
        try:
            for node in self.connections:
                # Select relevant patterns for the node
                relevant_patterns = [p for p in patterns 
                                   if self._is_pattern_relevant(p, node)]
                
                if relevant_patterns:
                    teaching = {
                        'target_node': node.id,
                        'patterns': relevant_patterns,
                        'timestamp': time.time()
                    }
                    teachings.append(teaching)
                    
                    # Share knowledge
                    node.receive_teaching(relevant_patterns)
                    
            return teachings
        except Exception:
            return teachings

    def receive_teaching(self, patterns: List[Dict]) -> None:
        """Process teaching received from another node"""
        try:
            # Verify and integrate received patterns
            valid_patterns = [p for p in patterns if self._verify_pattern(p)]
            self._learn_patterns(valid_patterns)
            
        except Exception as e:
            print(f"Teaching reception error: {e}")

    def _verify_pattern(self, pattern: Dict) -> bool:
        """Verify received pattern's validity"""
        try:
            # Basic verification
            required_keys = ['type', 'key', 'value']
            if not all(key in pattern for key in required_keys):
                return False
                
            # Check for suspicious patterns
            if any(self._is_pattern_suspicious(pattern)):
                return False
                
            return True
        except Exception:
            return False

    def _is_pattern_suspicious(self, pattern: Dict) -> bool:
        """Check for suspicious pattern characteristics"""
        try:
            # Check for abnormal values
            if isinstance(pattern.get('value'), (int, float)):
                if abs(pattern['value']) > 1e6:  # Arbitrary threshold
                    return True
                    
            # Check for suspicious strings
            if isinstance(pattern.get('value'), str):
                suspicious_terms = ['hack', 'exploit', 'override']
                if any(term in pattern['value'].lower() for term in suspicious_terms):
                    return True
                    
            return False
        except Exception:
            return True  # Consider suspicious if verification fails

