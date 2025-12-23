def __init__(self):
    self.graph = nx.DiGraph()

def add_insight(self, insight: Dict):
    node_id = str(uuid.uuid4())
    self.graph.add_node(node_id, **insight)
    # Connect to similar nodes (real logic: cosine > 0.5)
    for n in self.graph.nodes:
        if cosine(embed_text(insight['content']), embed_text(self.graph.nodes[n]['content'])) < 0.5:
            self.graph.add_edge(n, node_id)

def propagate(self):
    for node in list(self.graph.nodes):
        for succ in list(self.graph.successors(node)):
            # Share data (real merge)
            self.graph.nodes[succ]['content'] += " " + self.graph.nodes[node]['content']

