# knowledge_pool.py

from dataclasses import dataclass, field
from typing import Dict, Set, Optional
import uuid
import networkx as nx
import time

@dataclass
class ContextualPattern:
    content: Dict
    confidence: float = 0.5
    timestamp: float = field(default_factory=time.time)

class EnhancedSharedKnowledgePool:
    def __init__(self):
        self.patterns = {}
        self.pattern_graph = nx.Graph()

    def add_pattern(self, pattern: Dict):
        pattern_id = str(uuid.uuid4())
        self.patterns[pattern_id] = ContextualPattern(content=pattern)
        self.pattern_graph.add_node(pattern_id)

    def get_related_patterns(self, pattern_id: str) -> List[ContextualPattern]:
        neighbors = self.pattern_graph.neighbors(pattern_id)
        return [self.patterns[neighbor_id] for neighbor_id in neighbors if neighbor_id in self.patterns]

