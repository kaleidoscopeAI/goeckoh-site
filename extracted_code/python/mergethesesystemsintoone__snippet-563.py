def __init__(self):
    self.graph = nx.DiGraph()

def add_insight_batch(self, insights: List[Dict]):
    for ins in insights:
        node_id = ins.get('id', str(uuid.uuid4()))
        self.graph.add_node(node_id, **ins)
    embs = np.array([embed_text(ins.get('content', '')) for ins in insights])
    for i in range(len(insights)):
        for j in range(i+1, len(insights)):
            sim = 1 - math.cos(embs[i], embs[j])
            if sim > 0.5:
                self.graph.add_edge(insights[i]['id'], insights[j]['id'], weight=sim)

def propagate(self):
    order = list(nx.topological_sort(self.graph))
    for node in order:
        preds = list(self.graph.predecessors(node))
        if preds and 'content' in self.graph.nodes[node]:
            merged = " | ".join(self.graph.nodes[p].get('content', '')[:50] for p in preds)
            self.graph.nodes[node]['content'] += merged

def find_interventions(self):
    sample = random.sample(list(self.graph.nodes), min(1000, len(self.graph.nodes)))
    subgraph = self.graph.subgraph(sample)
    betweenness = nx.betweenness_centrality(subgraph, k=100)
    return sorted(betweenness.items(), key=lambda x: x[1], reverse=True)[:10]

