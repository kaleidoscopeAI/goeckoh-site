class SynapticOverlay:
    def __init__(self):
        self.overlay = {}

    def add_connection(self, source_node, target_node, context):
        """Add a synaptic connection between nodes."""
        if source_node not in self.overlay:
            self.overlay[source_node] = {}
        self.overlay[source_node][target_node] = context

    def visualize_overlay(self):
        """Visualize the synaptic overlay."""
        import networkx as nx
        import matplotlib.pyplot as plt

        graph = nx.DiGraph()
        for source, targets in self.overlay.items():
            for target, context in targets.items():
                graph.add_edge(source, target, label=context)

        pos = nx.spring_layout(graph)
        nx.draw(graph, pos, with_labels=True, node_color="lightblue", edge_color="grey", node_size=500, font_size=10)
        edge_labels = nx.get_edge_attributes(graph, 'label')
        nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels)
        plt.show()

