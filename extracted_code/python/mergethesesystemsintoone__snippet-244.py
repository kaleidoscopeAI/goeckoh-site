import networkx as nx
from typing import Dict, List, Optional, Any, Tuple
from collections import defaultdict
import numpy as np

class KnowledgeGraph:
    def __init__(self):
        self.graph = nx.DiGraph()  # Use a directed graph

    def add_node(self, node_id: str, data: Dict = None):
        """Adds a node to the knowledge graph."""
