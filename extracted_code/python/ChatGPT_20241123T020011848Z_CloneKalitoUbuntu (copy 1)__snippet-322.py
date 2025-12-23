def __init__(self):
    self.graph = nx.Graph()
    self.step = 0

def add_node(self, node_id):
    self.graph.add_node(node_id)

def add_edge(self, node1, node2):
    self.graph.add_edge(node1, node2)

def update(self):
    """Simulate network updates."""
    self.add_node(self.step)
    if self.step > 0:
        self.add_edge(self.step, self.step - 1)
    self.step += 1

def visualize(self):
    """Animate the network."""
    fig, ax = plt.subplots(figsize=(8, 6))
    ani = FuncAnimation(fig, self._animate, fargs=(ax,), interval=500, repeat=False)
    plt.show()

def _animate(self, frame, ax):
    self.update()
    ax.clear()  # Clear the previous frame
    nx.draw(self.graph, ax=ax, with_labels=True, node_size=500, font_size=10, node_color="lightgreen")

