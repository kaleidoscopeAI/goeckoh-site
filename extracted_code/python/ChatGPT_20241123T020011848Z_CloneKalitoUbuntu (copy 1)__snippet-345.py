def __init__(self):
    self.graph = nx.Graph()

def add_node(self, node_id):
    self.graph.add_node(node_id)

def visualize(self):
    nx.draw(self.graph, with_labels=True, node_color="lightblue")
    plt.show()

