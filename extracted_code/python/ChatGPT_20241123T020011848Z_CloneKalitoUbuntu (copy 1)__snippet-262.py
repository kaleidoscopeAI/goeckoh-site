def __init__(self):
    self.id = str(uuid.uuid4())
    self.birth_time = time.time()
    self.energy = 1.0
    self.growth_state = {
        'maturity': 0.0,
        'knowledge': 0.0,
        'specialization': None
    }

    # Core traits that influence behavior
    self.traits = {
        'learning_rate': 0.01,
        'adaptation_rate': 0.01,
        'energy_efficiency': 1.0
    }

    # Memory systems
    self.short_term = []
    self.long_term = {}

    # Growth tracking
    self.experiences = []
    self.growth_path = []

def process_input(self, data: Dict) -> Dict:
    """Process input data and grow from it"""
    try:
        # Extract patterns
        patterns = self._extract_patterns(data)

        # Learn from patterns
        learning_result = self._learn_from_patterns(patterns)

        # Update growth state
        self._update_growth_state(learning_result)

        # Track experience
        self.experiences.append({
            'input': data,
            'patterns': patterns,
            'learning': learning_result,
            'timestamp': time.time()
        })

        return {
            'processed': True,
            'patterns_found': len(patterns),
            'learning_gain': learning_result['knowledge_gain']
        }

    except Exception as e:
        print(f"Processing error: {str(e)}")
        return {'processed': False, 'error': str(e)}

def _extract_patterns(self, data: Dict) -> List[Dict]:
    """Extract meaningful patterns from input data"""
    patterns = []

    # Process text patterns if present
    if 'text' in data:
        text_patterns = self._process_text(data['text'])
        patterns.extend(text_patterns)

    # Process numerical patterns if present
    if 'numbers' in data:
        num_patterns = self._process_numbers(data['numbers'])
        patterns.extend(num_patterns)

    # Filter significant patterns
    return [p for p in patterns if p['significance'] > 0.3]

def _process_text(self, text: str) -> List[Dict]:
    """Process text data for patterns"""
    patterns = []
    words = text.split()

    # Word frequency analysis
    word_freq = {}
    for word in words:
        if len(word) > 3:  # Ignore small words
            word_freq[word] = word_freq.get(word, 0) + 1

    # Extract significant patterns
    for word, freq in word_freq.items():
        if freq > 1:  # Must appear more than once
            patterns.append({
                'type': 'text',
                'content': word,
                'frequency': freq,
                'significance': freq / len(words)
            })

    return patterns

def _process_numbers(self, numbers: List[float]) -> List[Dict]:
    """Process numerical data for patterns"""
    patterns = []

    if len(numbers) < 2:
        return patterns

    # Statistical patterns
    mean = np.mean(numbers)
    std = np.std(numbers)

    patterns.append({
        'type': 'numerical',
        'mean': mean,
        'std': std,
        'range': (min(numbers), max(numbers)),
        'significance': 1 / (1 + std)  # Higher significance for more consistent patterns
    })

    return patterns

def _learn_from_patterns(self, patterns: List[Dict]) -> Dict:
    """Learn from discovered patterns"""
    knowledge_gain = 0

    for pattern in patterns:
        # Calculate learning impact
        impact = pattern['significance'] * self.traits['learning_rate']
        knowledge_gain += impact

        # Store in memory
        if pattern['significance'] > 0.5:
            self.long_term[pattern['content']] = {
                'pattern': pattern,
                'impact': impact,
                'timestamp': time.time()
            }
        else:
            self.short_term.append({
                'pattern': pattern,
                'impact': impact,
                'timestamp': time.time()
            })

    return {
        'knowledge_gain': knowledge_gain,
        'patterns_stored': len(patterns)
    }

def _update_growth_state(self, learning_result: Dict):
    """Update node's growth state"""
    # Update knowledge
    self.growth_state['knowledge'] += learning_result['knowledge_gain']

    # Update maturity based on knowledge
    self.growth_state['maturity'] = min(1.0, 
        self.growth_state['maturity'] + 
        learning_result['knowledge_gain'] * self.traits['adaptation_rate']
    )

    # Update energy
    energy_cost = learning_result['knowledge_gain'] * (1 / self.traits['energy_efficiency'])
    self.energy = max(0, self.energy - energy_cost)

    # Track growth
    self.growth_path.append({
        'knowledge': self.growth_state['knowledge'],
        'maturity': self.growth_state['maturity'],
        'energy': self.energy,
        'timestamp': time.time()
    })

