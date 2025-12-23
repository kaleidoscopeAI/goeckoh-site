def __init__(self, node_id=None):
    self.id = node_id or str(uuid.uuid4())
    self.birth_time = time.time()
    self.energy = 10.0
    self.growth_state = {'maturity': 0.0, 'knowledge': 0.0, 'specialization': None}
    self.traits = {'learning_rate': 0.1, 'adaptation_rate': 0.1, 'energy_efficiency': 1.0}
    self.short_term = []
    self.long_term = {}
    self.experiences = []
    self.connections = set()

def process_input(self, data: Dict) -> Dict:
    patterns = self._extract_patterns(data)
    learning_result = self._learn_from_patterns(patterns)
    self._update_growth_state(learning_result)
    self._share_knowledge()
    self.experiences.append({'input': data, 'patterns': patterns, 'learning': learning_result, 'timestamp': time.time()})
    if 'text' in data: self.growth_state['specialization'] = 'text_analysis'
    elif 'numbers' in data: self.growth_state['specialization'] = 'numerical_analysis'
    return {'processed': True, 'patterns_found': len(patterns), 'learning_gain': learning_result['knowledge_gain']}

def _extract_patterns(self, data: Dict) -> List[Dict]:
    patterns = []
    if 'text' in data: patterns.extend(self._process_text(data['text']))
    if 'numbers' in data: patterns.extend(self._process_numbers(data['numbers']))
    return [p for p in patterns if p['significance'] > 0.3]

def _process_text(self, text: str) -> List[Dict]:
    patterns = []
    vectorizer = CountVectorizer(stop_words='english')
    word_freq_matrix = vectorizer.fit_transform([text])
    word_freq = word_freq_matrix.toarray().flatten()
    pca = PCA(n_components=1)
    pca_result = pca.fit_transform(word_freq_matrix.toarray())
    significance = abs(pca_result[0][0])
    if significance > 0.5: patterns.append({'type': 'text', 'significance': significance})
    return patterns

def _process_numbers(self, numbers: List[float]) -> List[Dict]:
    if len(numbers) < 2: return []
    mean, std = np.mean(numbers), np.std(numbers)
    return [{'type': 'numerical', 'mean': mean, 'std': std, 'range': (min(numbers), max(numbers)), 'significance': 1 / (1 + std)}]

def _learn_from_patterns(self, patterns: List[Dict]) -> Dict:
    knowledge_gain = 0
    for pattern in patterns:
        impact = pattern['significance'] * self.traits['learning_rate']
        knowledge_gain += impact
        memory_entry = {'pattern': pattern, 'impact': impact, 'timestamp': time.time()}
        if pattern['significance'] > 0.5: self.long_term[pattern['type']] = memory_entry
        else: self.short_term.append(memory_entry)
    return {'knowledge_gain': knowledge_gain, 'patterns_stored': len(patterns)}

def _update_growth_state(self, learning_result: Dict):
    self.growth_state['knowledge'] += learning_result['knowledge_gain']
    self.growth_state['maturity'] = min(1.0, self.growth_state['maturity'] + learning_result['knowledge_gain'] * self.traits['adaptation_rate'])
    energy_cost = learning_result['knowledge_gain'] * (1 / self.traits['energy_efficiency'])
    self.energy = max(0, self.energy - energy_cost)

def _share_knowledge(self):
    for node in self.connections:
        for key, memory in self.long_term.items():
            if key not in node.long_term:
                node.long_term[key] = memory

def replicate(self):
    new_node = Node()
    for trait in self.traits:
        mutation = np.random.normal(0, 0.01)
        new_node.traits[trait] = max(0.01, self.traits[trait] + mutation)
    for key, memory in np.random.choice(list(self.long_term.items()), size=min(3, len(self.long_term)), replace=False):
        new_node.long_term[key] = memory
    return new_node

