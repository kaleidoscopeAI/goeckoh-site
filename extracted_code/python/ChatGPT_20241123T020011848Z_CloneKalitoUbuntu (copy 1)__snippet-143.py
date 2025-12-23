import numpy as np
import time
import math
import uuid
from typing import Dict, List, Optional

class MathematicalModel:
    """
    Implements the mathematical models from the specifications
    """
    @staticmethod
    def node_growth_rate(N0: float, r: float, t: float) -> float:
        """N(t) = N0 * e^(rt): Node growth model"""
        return N0 * math.exp(r * t)
    
    @staticmethod
    def knowledge_growth(K0: float, k: float, c: float, t: float) -> float:
        """K(t) = K0 * e^((k+c)t): Knowledge accumulation"""
        return K0 * math.exp((k + c) * t)
    
    @staticmethod
    def system_failure_probability(p: float, n: int) -> float:
        """P(system failure) = p^n: Resilience calculation"""
        return p ** n
    
    @staticmethod
    def parallel_processing_time(Ts: float, P: float, N: int) -> float:
        """T_p = T_s * ((1 - P) + (P / N)): Amdahl's Law"""
        return Ts * ((1 - P) + (P / N))
    
    @staticmethod
    def learning_efficiency(E0: float, R: float, t: float) -> float:
        """E(t) = E0 * ln(1 + Rt): Learning convergence"""
        return E0 * math.log(1 + R * t)

class EnhancedNodeDNA:
    """
    Enhanced DNA with mathematical models and self-reference
    """
    def __init__(self, traits: Dict[str, float]):
        self.id = str(uuid.uuid4())
        self.creation_time = time.time()
        self.traits = traits
        self.math_model = MathematicalModel()
        
        # Growth parameters
        self.learning_rate = 0.01  # k in knowledge growth formula
        self.cross_learning_rate = 0.005  # c in knowledge growth formula
        self.initial_knowledge = 1.0  # K0 in knowledge growth formula
        
        # Evolution tracking
        self.mutations = []
        self.evolution_history = []
        
    def calculate_current_knowledge(self) -> float:
        """Calculate current knowledge level using compound growth formula"""
        t = time.time() - self.creation_time
        return self.math_model.knowledge_growth(
            self.initial_knowledge,
            self.learning_rate,
            self.cross_learning_rate,
            t
        )
    
    def mutate(self) -> 'EnhancedNodeDNA':
        """Create evolved DNA with tracked mutations"""
        new_traits = {}
        mutation_record = {
            'time': time.time(),
            'changes': []
        }
        
        for trait, value in self.traits.items():
            # Calculate mutation based on current knowledge
            knowledge_factor = self.calculate_current_knowledge()
            mutation = np.random.normal(0, 0.1) * knowledge_factor
            
            new_value = max(0, value + mutation)
            new_traits[trait] = new_value
            
            mutation_record['changes'].append({
                'trait': trait,
                'from': value,
                'to': new_value,
                'factor': knowledge_factor
            })
            
        # Create new DNA
        new_dna = EnhancedNodeDNA(new_traits)
        new_dna.mutations = self.mutations + [mutation_record]
        
        return new_dna
    
    def evolve_from_experiences(self, experiences: List) -> None:
        """Evolve traits based on accumulated experiences"""
        evolution_record = {
            'time': time.time(),
            'experience_count': len(experiences),
            'changes': []
        }
        
        for trait, value in self.traits.items():
            # Calculate evolution based on experiences and current knowledge
            experience_impact = sum(exp.get('impact', 0) for exp in experiences)
            knowledge_level = self.calculate_current_knowledge()
            
            # Apply logarithmic learning curve
            adjustment = self.math_model.learning_efficiency(
                E0=0.1,  # Base efficiency
                R=abs(experience_impact),  # Impact as reinforcement rate
                t=knowledge_level  # Current knowledge as time factor
            )
            
            old_value = value
            self.traits[trait] = max(0, value + adjustment)
            
            evolution_record['changes'].append({
                'trait': trait,
                'from': old_value,
                'to': self.traits[trait],
                'adjustment': adjustment
            })
            
        self.evolution_history.append(evolution_record)

class EnhancedOrganicCore:
    """
    Enhanced core node with mathematical models and self-reference
    """
    def __init__(self, node_id: str, dna: Optional[EnhancedNodeDNA] = None):
        self.node_id = node_id
        self.birth_time = time.time()
        self.dna = dna or self._initialize_dna()
        
        # Core state
        self.memory = []
        self.energy = 1.0
        self.connections = set()
        
        # Growth tracking
        self.children = []
        self.growth_rate = 0.1  # r in node growth formula
        self.parallel_factor = 0.8  # P in Amdahl's Law
        
        # Mathematical model
        self.math = MathematicalModel()
        
    def _initialize_dna(self) -> EnhancedNodeDNA:
        """Initialize DNA with base traits"""
        return EnhancedNodeDNA({
            'learning_capacity': 1.0,
            'adaptation_rate': 1.0,
            'resilience': 1.0,
            'efficiency': 1.0,
            'specialization': 1.0
        })
        
    def calculate_growth_potential(self) -> float:
        """Calculate growth potential using node growth model"""
        t = time.time() - self.birth_time
        N0 = 1.0  # Single node
        return self.math.node_growth_rate(N0, self.growth_rate, t)
    
    def calculate_resilience(self) -> float:
        """Calculate system resilience"""
        base_failure_prob = 0.1
        redundancy = len(self.connections) + 1
        return 1 - self.math.system_failure_probability(base_failure_prob, redundancy)
    
    def process_experience(self, experience: Dict) -> Dict:
        """Process experience with parallel efficiency"""
        processing_time = 1.0  # Base processing time
        node_count = len(self.connections) + 1
        
        # Calculate parallel processing time
        actual_time = self.math.parallel_processing_time(
            processing_time,
            self.parallel_factor,
            node_count
        )
        
        # Process experience
        self.memory.append({
            'data': experience,
            'processing_time': actual_time,
            'timestamp': time.time()
        })
        
        # Learn from experience
        self.dna.evolve_from_experiences([experience])
        
        return {
            'processed': True,
            'time_taken': actual_time,
            'knowledge_gain': self.dna.calculate_current_knowledge()
        }
