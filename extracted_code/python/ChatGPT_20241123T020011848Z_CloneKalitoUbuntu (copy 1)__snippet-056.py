import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

class MirroredNetwork:
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

        def _animate(frame):
            ax.clear()  # Clear the previous frame
            self.update()
            nx.draw(self.graph, ax=ax, with_labels=True, node_size=500, font_size=10, node_color="lightgreen")

        ani = FuncAnimation(fig, _animate, frames=20, interval=500, repeat=False)
        plt.show()

if __name__ == "__main__":
    network = MirroredNetwork()
    network.visualize()

