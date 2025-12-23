def __init__(self):
    self.graph = nx.DiGraph()  # Directed graph for hierarchical flow
    self.nodes = []
    self.step = 0

def add_node(self, word):
    node = MirroredNode(self.step, word)
    node.process_textual()
    node.process_visual()
    node.combine_results()
    self.graph.add_node(self.step, word=node.word, context=node.final_context)
    self.nodes.append(node)
    self.step += 1

def connect_nodes(self):
    """Simulate hierarchical relationships."""
    if len(self.nodes) > 1:
        self.graph.add_edge(self.step - 2, self.step - 1)  # Connect parent to child

def visualize(self):
    """Visualize the network."""
    fig, ax = plt.subplots(figsize=(10, 6))
    pos = nx.spring_layout(self.graph)

    def _animate(frame):
        if frame < len(self.nodes):
            self.add_node(random.choice(["hotdog", "cat", "hotdog", "tree"]))
            self.connect_nodes()
            ax.clear()
            nx.draw(self.graph, pos, ax=ax, with_labels=True, node_size=700, node_color="lightblue", font_size=8)

    ani = FuncAnimation(fig, _animate, frames=10, interval=1000, repeat=False)
    plt.show()

