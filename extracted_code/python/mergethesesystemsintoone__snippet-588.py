def __init__(self):
    self.graph = nx.DiGraph()  # Use a directed graph

def add_node(self, node_id: str, data: Dict = None):
    """Adds a node to the knowledge graph."""
    if node_id not in self.graph:
        self.graph.add_node(node_id, data=data if data else {})

def add_edge(self, node1: str, node2: str, relationship: Dict = None):
    """Adds an edge between two nodes in the knowledge graph."""
    if node1 in self.graph and node2 in self.graph:
        if not self.graph.has_edge(node1, node2):
            self.graph.add_edge(node1, node2, **relationship if relationship else {})
        else:
            # Update the existing edge data
            self.graph[node1][node2].update(relationship if relationship else {})

def get_node_data(self, node_id: str) -> Dict:
    """Retrieves data associated with a node."""
    if node_id in self.graph.nodes:
        return self.graph.nodes[node_id]['data']
    return {}

def get_related_nodes(self, node_id: str, relationship_type: Optional[str] = None) -> List[str]:
    """Finds nodes related to a given node, optionally filtered by relationship type."""
    related_nodes = []
    if node_id in self.graph:
        for neighbor in self.graph.neighbors(node_id):
            if relationship_type:
                # Check if the relationship type matches
                edge_data = self.graph.get_edge_data(node_id, neighbor)
                if edge_data.get('type') == relationship_type:
                    related_nodes.append(neighbor)
            else:
                related_nodes.append(neighbor)
    return related_nodes

def update_node_data(self, node_id: str, data: Dict):
    """Updates the data associated with a node."""
    if node_id in self.graph.nodes:
        self.graph.nodes[node_id]['data'].update(data)

def update_edge_data(self, node1: str, node2: str, data: Dict):
    """Updates the data associated with an edge."""
    if self.graph.has_edge(node1, node2):
        self.graph[node1][node2].update(data)

def extract_concepts(self, data: Dict) -> List[Dict]:
    """
    Extracts concepts from data and adds them to the knowledge graph.
    """
    concepts = []

    # Example: Extract concepts from text patterns
    text_patterns = data.get('text_patterns', [])
    for pattern in text_patterns:
        if pattern['type'] == 'named_entity':
            entity = pattern['entity']
            self.add_node(entity, {'type': 'named_entity', 'label': pattern['label']})
            concepts.append({'id': entity, 'type': 'named_entity'})
        elif pattern['type'] == 'word_embedding':
            self.add_node(pattern['word'],{'type': 'word_embedding'})
            concepts.append({'id': pattern['word'], 'type': 'word_embedding'})

    # Example: Extract concepts from visual patterns
    visual_patterns = data.get('visual_patterns', [])
    for pattern in visual_patterns:
        if pattern['type'] == 'shape':
            shape_type = pattern['shape_type']
            self.add_node(shape_type, {'type': 'shape', 'vertices': pattern['vertices']})
            concepts.append({'id': shape_type, 'type': 'shape'})
        elif pattern['type'] == 'color_patterns':
          for color_pattern in pattern['dominant_colors']:
            self.add_node(str(color_pattern['color']), {'type': 'color', 'frequency': color_pattern['frequency']})
            concepts.append({'id': str(color_pattern['color']), 'type': 'color'})

    return concepts

def find_related_concepts(self, concept: str, depth: int = 2) -> List[Tuple[str, float]]:
    """
    Finds concepts related to the given concept in the knowledge graph using BFS.
    """
    related_concepts = []
    if concept in self.graph:
        # Perform a breadth-first search to find related concepts within the specified depth
        bfs_tree = nx.bfs_tree(self.graph, source=concept, depth_limit=depth)
        for neighbor in bfs_tree.nodes():
            if neighbor != concept:
                # Calculate a relevance score based on the path length
                try:
                    path_length = nx.shortest_path_length(self.graph, source=concept, target=neighbor)
                    relevance_score = 1 / path_length if path_length > 0 else 0
                    related_concepts.append((neighbor, relevance_score))
                except nx.NetworkXNoPath:
                    logger.info(f"No path found between {concept} and {neighbor}")
    return related_concepts

def calculate_centrality(self, method: str = 'degree') -> Dict[str, float]:
    """
    Calculates the centrality of nodes in the knowledge graph.
    """
    if method == 'degree':
        centrality = nx.degree_centrality(self.graph)
    elif method == 'betweenness':
        centrality = nx.betweenness_centrality(self.graph)
    elif method == 'closeness':
        centrality = nx.closeness_centrality(self.graph)
    elif method == 'eigenvector':
        centrality = nx.eigenvector_centrality(self.graph, max_iter=1000)
    else:
        raise ValueError(f"Invalid centrality method: {method}")
    return centrality

def find_shortest_path(self, concept1: str, concept2: str) -> List[str]:
    """
    Finds the shortest path between two concepts in the knowledge graph.
    """
    if concept1 in self.graph and concept2 in self.graph:
        try:
            path = nx.shortest_path(self.graph, source=concept1, target=concept2)
            return path
        except nx.NetworkXNoPath:
            logger.info(f"No path found between {concept1} and {concept2}")
            return []
    else:
        logger.warning(f"One or both concepts not found in the knowledge graph: {concept1}, {concept2}")
        return []

