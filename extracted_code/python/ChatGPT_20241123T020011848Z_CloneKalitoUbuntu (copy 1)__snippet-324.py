def __init__(self):
    self.left_strand = nx.Graph()
    self.right_strand = nx.Graph()
    self.shared_hub = []
    self.step = 0

def add_node(self, node_id, strand="left"):
    if strand == "left":
        self.left_strand.add_node(node_id)
    else:
        self.right_strand.add_node(node_id)

def add_edge(self, node1, node2, strand="left"):
    if strand == "left":
        self.left_strand.add_edge(node1, node2)
    else:
        self.right_strand.add_edge(node1, node2)

def update(self):
    """Simulate network updates and information sharing."""
    # Update left strand
    self.add_node(self.step, "left")
    if self.step > 0:
        self.add_edge(self.step, self.step - 1, "left")

    # Update right strand (mirroring left with a slight delay)
    self.add_node(self.step, "right")
    if self.step > 0:
        self.add_edge(self.step, self.step - 1, "right")

    # Share information via the hub
    shared_data = f"Node {self.step} data"
    self.shared_hub.append(shared_data)
    print(f"Shared Hub Updated: {self.shared_hub}")

    self.step += 1

def visualize(self):
    """Animate the mirrored DNA network."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))

    def _animate(frame):
        ax.clear()
        self.update()

        # Draw the strands
        nx.draw(self.left_strand, ax=ax, pos=nx.spring_layout(self.left_strand), with_labels=True, node_size=700, node_color="blue", edge_color="lightblue")
        nx.draw(self.right_strand, ax=ax, pos=nx.spring_layout(self.right_strand), with_labels=True, node_size=700, node_color="green", edge_color="lightgreen")

        # Simulate the shared hub
        ax.set_title(f"Mirrored DNA Simulation - Step {frame}")

    ani = FuncAnimation(fig, _animate, frames=50, interval=500, repeat=False)
    plt.show()

