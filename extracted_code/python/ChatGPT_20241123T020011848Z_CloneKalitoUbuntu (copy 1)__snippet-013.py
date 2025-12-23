# ai_network_demo.py

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import uuid
import time
from typing import Dict, List

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
        # Extract patterns
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
        
        return {
            'processed': True,
            'patterns_found': len(patterns),
            'learning_gain': learning_result['knowledge_gain']
        }

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
        words = text.lower().split()
        
        # Word frequency analysis
        word_freq = {}
        for word in words:
            if len(word) > 3:  # Ignore small words
                word_freq[word] = word_freq.get(word, 0) + 1
                
        # Extract significant patterns
        total_words = len(words)
        for word, freq in word_freq.items():
            significance = freq / total_words
            if freq > 1:  # Must appear more than once
                patterns.append({
                    'type': 'text',
                    'content': word,
                    'frequency': freq,
                    'significance': significance
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
            memory_entry = {
                'pattern': pattern,
                'impact': impact,
                'timestamp': time.time()
            }
            if pattern['significance'] > 0.5:
                self.long_term[pattern['content']] = memory_entry
            else:
                self.short_term.append(memory_entry)
                
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

    def _share_knowledge(self):
        """Share knowledge with connected nodes"""
        for node in self.connections:
            # Share long-term memory patterns
            for key, memory in self.long_term.items():
                if key not in node.long_term:
                    node.long_term[key] = memory

    def replicate(self):
        """Replicate node with slight mutations"""
        new_node = Node()
        # Slightly mutate traits
        for trait in self.traits:
            mutation = np.random.normal(0, 0.01)
            new_node.traits[trait] = max(0.01, self.traits[trait] + mutation)
        # Inherit some long-term memory
        inherited_memory = np.random.choice(list(self.long_term.items()), 
                                            size=min(3, len(self.long_term)), 
                                            replace=False)
        for key, memory in inherited_memory:
            new_node.long_term[key] = memory
        return new_node

class Environment:
    def __init__(self):
        self.resources = 1000.0  # Total resources available
        self.nodes = []
        self.time = 0  # Simulation time

    def add_node(self, node: Node):
        self.nodes.append(node)

    def provide_resources(self, node: Node):
        """Provide resources to a node"""
        if self.resources > 0:
            resource_amount = 5.0  # Fixed resource allocation
            self.resources -= resource_amount
            node.energy += resource_amount * node.traits['energy_efficiency']
        else:
            node.energy -= 0.1  # Penalty if no resources available

    def simulate(self):
        """Simulate environment interactions"""
        for node in list(self.nodes):  # Use a copy of the list
            # Provide resources
            self.provide_resources(node)

            # Node processes input (simulate input data)
            input_data = self.generate_input_data()
            node.process_input(input_data)

            # Node may replicate
            if node.growth_state['maturity'] >= 1.0 and node.energy > 20.0:
                new_node = node.replicate()
                node.energy /= 2  # Energy cost for replication
                new_node.energy = node.energy
                self.add_node(new_node)
                # Connect nodes
                node.connections.add(new_node)
                new_node.connections.add(node)
                print(f"Node {node.id} replicated to create Node {new_node.id}")

            # Remove node if energy depleted
            if node.energy <= 0:
                self.nodes.remove(node)
                print(f"Node {node.id} has been removed due to energy depletion.")

    def generate_input_data(self) -> Dict:
        """Generate simulated input data"""
        # For simplicity, alternate between text and numerical data
        if self.time % 2 == 0:
            data = {
                'text': 'This is a sample text input for pattern recognition. Sample text input.'
            }
        else:
            data = {
                'numbers': np.random.normal(loc=50, scale=5, size=100).tolist()
            }
        self.time += 1
        return data

def visualize_network(nodes: List[Node]):
    """Visualize the node network"""
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
    plt.title('AI Node Network Visualization')
    plt.show()

# Simulation setup
environment = Environment()

# Create initial nodes
initial_node = Node(node_id='Node_0')
environment.add_node(initial_node)

# Run simulation cycles
for cycle in range(10):
    print(f"--- Cycle {cycle + 1} ---")
    environment.simulate()
    print(f"Total nodes: {len(environment.nodes)}")
    print(f"Total environment resources: {environment.resources}")
    # Visualize after each cycle
    visualize_network(environment.nodes)
    time.sleep(1)  # Pause for a second between cycles

