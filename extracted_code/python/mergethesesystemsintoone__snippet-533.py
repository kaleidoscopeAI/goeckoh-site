def __init__(self):
    self.graph = nx.DiGraph()

def add_insight(self, insight: Dict):
    node_id = str(uuid.uuid4())
    self.graph.add_node(node_id, **insight)
    for other in list(self.graph.nodes):
        sim = 1 - cosine(embed_text(insight.get('content', '')), embed_text(self.graph.nodes[other].get('content', '')))
        if sim > 0.5:
            self.graph.add_edge(other, node_id, weight=sim)

def propagate(self):
    for node in list(self.graph.nodes):
        for succ in list(self.graph.successors(node)):
            if 'content' in self.graph.nodes[succ] and 'content' in self.graph.nodes[node]:
                self.graph.nodes[succ]['content'] += " | " + self.graph.nodes[node]['content'][:100]  # Real merge

def find_interventions(self):
    betweenness = nx.betweenness_centrality(self.graph)
    return sorted(betweenness.items(), key=lambda x: x[1], reverse=True)[:5]  # Top 5 critical

