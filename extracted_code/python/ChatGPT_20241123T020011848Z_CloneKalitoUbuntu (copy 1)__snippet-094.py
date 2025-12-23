import matplotlib.pyplot as plt
import networkx as nx

def visualize_node_growth(nodes):
    graph = nx.Graph()
    for node in nodes:
        graph.add_node(node.node_id)
        for topic in node.knowledge:
            graph.add_edge(node.node_id, topic)

    nx.draw(graph, with_labels=True, node_color="skyblue", font_weight="bold")
    plt.show()

