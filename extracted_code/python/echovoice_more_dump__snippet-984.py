"""Simulates the 3D thought-form visualization from the React component."""
def __init__(self, num_nodes=8000):
    self.num_nodes = num_nodes
    self.target_positions = np.zeros((num_nodes, 3))
    print(f"Visualizer initialized for {num_nodes} particles.")

def create_visual_embedding(self, thought: str) -> np.ndarray:
    """Direct Python port of the createVisualEmbedding logic from JS."""
    thought_hash = sum(ord(c) for c in thought)

    # Create vectors based on the thought hash
    indices = np.arange(self.num_nodes)
    t = (indices / self.num_nodes) * np.pi * 8
    r = 300 + 100 * np.sin(thought_hash * 0.01 + indices * 0.02)
    twist = np.sin(thought_hash * 0.005) * 2.0

    x = r * np.cos(t + twist)
    y = r * np.sin(t + twist)
    z = 200 * np.sin(indices * 0.01 + thought_hash * 0.03)

    return np.stack([x, y, z], axis=1)

def update_from_thought(self, thought: str):
    """Updates the visual state and reports summary statistics."""
    self.target_positions = self.create_visual_embedding(thought)

    mins = self.target_positions.min(axis=0)
    maxs = self.target_positions.max(axis=0)
    avg_radius = np.mean(np.linalg.norm(self.target_positions, axis=1))

    print("\n--- Visualizer State Update ---")
    print(f"Thought-form embedded from text: '{thought[:60]}...'")
    print(f"Bounding Box (X, Y, Z):")
    print(f"  min: [{mins[0]:.1f}, {mins[1]:.1f}, {mins[2]:.1f}]")
    print(f"  max: [{maxs[0]:.1f}, {maxs[1]:.1f}, {maxs[2]:.1f}]")
    print(f"Average Particle Radius from Center: {avg_radius:.2f}")
    print("---------------------------------")


