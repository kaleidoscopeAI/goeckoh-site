import networkx as nx
import matplotlib.pyplot as plt

class MirroredNetwork:
    def __init__(self):
        self.graph = nx.Graph()
        self.nodes = {}

    def add_node(self, node):
        self.graph.add_node(node.node_id)
        self.nodes[node.node_id] = node

    def add_edge(self, node1_id, node2_id):
        self.graph.add_edge(node1_id, node2_id)

    def visualize(self):
        nx.draw(self.graph, with_labels=True, node_size=500, font_size=10, node_color="skyblue")
        plt.show()

    def sync_nodes(self, node1_id, node2_id):
        node1 = self.nodes[node1_id]
        node2 = self.nodes[node2_id]
        node1.communicate(node2)
        node2.communicate(node1)

