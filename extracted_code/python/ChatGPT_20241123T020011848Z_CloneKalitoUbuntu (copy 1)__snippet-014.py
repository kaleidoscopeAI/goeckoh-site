import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import uuid
import time
from typing import Dict, List
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import PCA

class Node:
    def __init__(self, node_id=None):
        self.id = node_id or str(uuid.uuid4())
        self.birth_time = time.time()
        self.energy = 10.0  # Starting energy
        self.growth_state = {
            'maturity': 0.0,
            'knowledge': 0.0,
            'specialization': None
        }
        
        # Core traits that influence behavior
        self.traits = {
            'learning_rate': 0.1,
            'adaptation_rate': 0.1,
            'energy_efficiency': 1.0
        }
        
        # Memory systems
        self.short_term = []
        self.long_term = {}
        
        # Growth tracking
        self.experiences = []
        self.growth_path = []
        
        # Connections to other nodes
        self.connections = set()

    def process_input(self, data: Dict) -> Dict:
        """Process input data and grow from it"""
        # Extract patterns using enhanced pattern recognition
        patterns = self._extract_patterns(data)
        
        # Learn from patterns
        learning_result = self._learn_from_patterns(patterns)
        
        # Update growth state
        self._update_growth_state(learning_result)
        
        # Share knowledge with connected nodes
        self._share_knowledge()
        
        # Track experience
        self.experiences.append({
            'input': data,
            'patterns': patterns,
            'learning': learning_result,
            'timestamp': time.time()
        })
        
        # Specialize based on input type (text or numerical)
        if 'text' in data:
            self.growth_state['specialization'] = 'text_analysis'
        elif 'numbers' in data:
            self.growth_state['specialization'] = 'numerical_analysis'
            
        return {
            'processed': True,
            'patterns_found': len(patterns),
            'learning_gain': learning_result['knowledge_gain']
        }

    def _extract_patterns(self, data: Dict) -> List[Dict]:
        """Extract meaningful patterns from input data with advanced recognition"""
        patterns = []
        
        if 'text' in data:
            text_patterns = self._process_text(data['text'])
            patterns.extend(text_patterns)
            
        if 'numbers' in data:
            num_patterns = self._process_numbers(data['numbers'])
            patterns.extend(num_patterns)
            
        return [p for p in patterns if p['significance'] > 0.3]

    def _process_text(self, text: str) -> List[Dict]:
        """Advanced text processing with machine learning techniques"""
        patterns = []
        
        # Use CountVectorizer for word frequency analysis
        vectorizer = CountVectorizer(stop_words='english')
        word_freq_matrix = vectorizer.fit_transform([text])
        word_freq = word_freq_matrix.toarray().flatten()
        
        # Apply PCA for pattern significance
        pca = PCA(n_components=1)
        pca_result = pca.fit_transform(word_freq_matrix.toarray())
        significance = abs(pca_result[0][0])
        
        if significance > 0.5:
            patterns.append({
                'type': 'text',
                'significance': significance
            })
        
        return patterns

    def _process_numbers(self, numbers: List[float]) -> List[Dict]:
        """Process numerical data for patterns"""
        patterns = []
        
        if len(numbers) < 2:
            return patterns
            
        mean = np.mean(numbers)
        std = np.std(numbers)
        
        patterns.append({
            'type': 'numerical',
            'mean': mean,
            'std': std,
            'range': (min(numbers), max(numbers)),
            'significance': 1 / (1 + std)
        })
        
        return patterns

    def _learn_from_patterns(self, patterns: List[Dict]) -> Dict:
        """Learn from discovered patterns with added complexity"""
        knowledge_gain = 0
        
        for pattern in patterns:
            impact = pattern['significance'] * self.traits['learning_rate']
            knowledge_gain += impact
            
            memory_entry = {
                'pattern': pattern,
                'impact': impact,
                'timestamp': time.time()
            }
            if pattern['significance'] > 0.5:
                self.long_term[pattern['type']] = memory_entry
            else:
                self.short_term.append(memory_entry)
                
        return {
            'knowledge_gain': knowledge_gain,
            'patterns_stored': len(patterns)
        }

    def _update_growth_state(self, learning_result: Dict):
        """Update node's growth state"""
        self.growth_state['knowledge'] += learning_result['knowledge_gain']
        self.growth_state['maturity'] = min(1.0, 
            self.growth_state['maturity'] + 
            learning_result['knowledge_gain'] * self.traits['adaptation_rate']
        )
        energy_cost = learning_result['knowledge_gain'] * (1 / self.traits['energy_efficiency'])
        self.energy = max(0, self.energy - energy_cost)
        
        self.growth_path.append({
            'knowledge': self.growth_state['knowledge'],
            'maturity': self.growth_state['maturity'],
            'energy': self.energy,
            'timestamp': time.time()
        })

    def _share_knowledge(self):
        """Share knowledge with connected nodes"""
        for node in self.connections:
            for key, memory in self.long_term.items():
                if key not in node.long_term:
                    node.long_term[key] = memory

    def replicate(self):
        """Replicate node with slight mutations"""
        new_node = Node()
        for trait in self.traits:
            mutation = np.random.normal(0, 0.01)
            new_node.traits[trait] = max(0.01, self.traits[trait] + mutation)
        inherited_memory = np.random.choice(list(self.long_term.items()), 
                                            size=min(3, len(self.long_term)), 
                                            replace=False)
        for key, memory in inherited_memory:
            new_node.long_term[key] = memory
        return new_node

class Environment:
    def __init__(self):
        self.resources = 1000.0
        self.nodes = []
        self.time = 0

    def add_node(self, node: Node):
        self.nodes.append(node)

    def provide_resources(self, node: Node):
        """Provide resources based on dynamic events"""
        event = self.simulate_event()
        resource_amount = 5.0 + event.get('bonus', 0)
        if self.resources > 0:
            self.resources -= resource_amount
            node.energy += resource_amount * node.traits['energy_efficiency']
        else:
            node.energy -= 0.1

    def simulate_event(self):
        """Simulate environmental events (scarcity/surplus)"""
        event_type = np.random.choice(['normal', 'scarcity', 'surplus'], p=[0.6, 0.2, 0.2])
        if event_type == 'scarcity':
            return {'event': 'scarcity', 'bonus': -3}
        elif event_type == 'surplus':
            return {'event': 'surplus', 'bonus': 3}
        else:
            return {'event': 'normal', 'bonus': 0}

    def simulate(self):
        for node in list(self.nodes):
            self.provide_resources(node)
            input_data = self.generate_input_data()
            node.process_input(input_data)
            if node.growth_state['maturity'] >= 1.0 and node.energy > 20.0:
                new_node = node.replicate()
                node.energy /= 2
                new_node.energy = node.energy
                self.add_node(new_node)
                node.connections.add(new_node)
                new_node.connections.add(node)
            if node.energy <= 0:
                self.nodes.remove(node)

    def generate_input_data(self) -> Dict:
        if self.time % 2 == 0:
            data = {'text': 'Custom input for advanced AI demo.'}
        else:
            data = {'numbers': np.random.normal(50, 5, 100).tolist()}
        self.time += 1
        return data

def visualize_network(nodes: List[Node]):
    G = nx.Graph()
    for node in nodes:
        G.add_node(node.id, energy=node.energy)
        for connected_node in node.connections:
            G.add_edge(node.id, connected_node.id)

    pos = nx.spring_layout(G)
    energies = [G.nodes[n]['energy'] for n in G.nodes()]
    max_energy = max(energies) if energies else 1
    node_sizes = [max(100, e / max_energy * 300) for e in energies]
    nx.draw(G, pos, with_labels=True, node_size=node_sizes, node_color=energies, cmap=plt.cm.viridis)
    plt.title('Enhanced AI Network Visualization')
    plt

