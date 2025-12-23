from perspective_engine import PerspectiveEngine
from emotional_transformer import EmotionalTransformer
from knowledge_graph import KnowledgeGraph
from complete_node import CompleteNode
from core_math import Vector

class KaleidoscopeEngine:
    def __init__(self, nodes, grid, rng, r_dim=3, dt=0.01, seed=None):
        self.nodes = nodes
        self.grid = grid
        self.rng = rng
        self.dt = dt
        self.e8_lattice = E8Lattice(seed=self.rng.randint(0, 100000))
        self.perspective_engine = PerspectiveEngine(self.e8_lattice, self.rng)
        self.emotional_transformer = EmotionalTransformer()
        self.knowledge_graph = KnowledgeGraph()
        self.crystallization_threshold = 0.8

    def run_cycle(self, step, input_texts):
        for node in self.nodes:
            if step % 10 == 0: 
                hypo = self.perspective_engine.generate_hypothesis(node.position)
                confidence = self.perspective_engine.evaluate_hypothesis(node.position, hypo,
                                            node.energy, node.knowledge, node.emotional_state)
                if confidence > 0.5:
                    node.position = hypo
                    node.knowledge = min(1.0, node.knowledge + confidence * 0.1)

        for node, text in zip(self.nodes, input_texts):
            out = self.emotional_transformer.forward_with_emotion(text, node.emotional_state)
            knowledge_boost = sum(out) * 0.01
            node.knowledge = min(1.0, node.knowledge + knowledge_boost)
            if node.knowledge > self.crystallization_threshold:
                symbolic_data = redact_pii(text)
                self.knowledge_graph.update_node_attributes(node.id, {
                    'E': node.energy,
                    'A': node.awareness,
                    'K': node.knowledge,
                    'position': node.position.components,
                    'symbolic_data': symbolic_data
                })
