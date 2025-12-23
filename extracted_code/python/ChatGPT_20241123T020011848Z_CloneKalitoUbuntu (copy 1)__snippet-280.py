"""Enhanced shared knowledge pool with advanced pattern relationships"""
def __init__(self):
    self.patterns = {}
    self.pattern_graph = nx.Graph()
    self.pattern_clusters = {}
    self.access_history = defaultdict(list)

def add_pattern(self, pattern: Dict, confidence: float):
    """Add pattern to pool with relationship mapping"""
    pattern_id = str(uuid.uuid4())
    self.patterns[pattern_id] = {
        'content': pattern,
        'confidence': confidence,
        'timestamp': time.time(),
        'access_count': 0
    }

    # Update pattern relationships
    self._update_pattern_relationships(pattern_id)

    # Update clusters
    self._update_pattern_clusters()

def _update_pattern_relationships(self, new_pattern_id: str):
    """Update pattern relationships using advanced metrics"""
    new_pattern = self.patterns[new_pattern_id]
    self.pattern_graph.add_node(new_pattern_id)

    # Calculate relationships with existing patterns
    for pid, pattern in self.patterns.items():
        if pid != new_pattern_id:
            relationship_strength = self._calculate_relationship_strength(
                new_pattern['content'],
                pattern['content']
            )

            if relationship_strength > 0.5:
                self.pattern_graph.add_edge(
                    new_pattern_id,
                    pid,
                    weight=relationship_strength
                )

def _update_pattern_clusters(self):
    """Update pattern clusters using DBSCAN"""
    if len(self.patterns) < 2:
        return

    # Extract pattern features
    features = self._extract_pattern_features()

    # Normalize features
    scaler = StandardScaler()
    normalized_features = scaler.fit_transform(features)

    # Perform clustering
    clustering = DBSCAN(eps=0.3, min_samples=2)
    cluster_labels = clustering.fit_predict(normalized_features)

    # Update cluster assignments
    for pattern_id, cluster_label in zip(self.patterns.keys(), cluster_labels):
        if cluster_label >= 0:  # Skip noise points
            self.pattern_clusters[pattern_id] = cluster_label

