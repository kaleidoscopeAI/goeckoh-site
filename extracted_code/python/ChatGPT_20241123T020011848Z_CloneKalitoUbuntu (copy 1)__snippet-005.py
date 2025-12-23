import matplotlib.pyplot as plt
import random

# Set up figure
plt.figure(figsize=(8, 8))
plt.title("Seed-Based Node Growth Simulation")
plt.xlim(-10, 10)
plt.ylim(-10, 10)

# Starting position of the seed node
nodes = [(0, 0)]  # Seed node at the center

# Parameters
replications = 8  # Number of replication rounds
offset_range = 1  # Distance from parent node for each replication

# Simulation of node replication with basic adaptation
for _ in range(replications):
    new_nodes = []
    for x, y in nodes:
        # Create two new nodes for each existing node, with slight random offsets
        new_x1, new_y1 = x + random.uniform(-offset_range, offset_range), y + random.uniform(-offset_range, offset_range)
        new_x2, new_y2 = x + random.uniform(-offset_range, offset_range), y + random.uniform(-offset_range, offset_range)
        
        # Append new nodes to the list
        new_nodes.extend([(new_x1, new_y1), (new_x2, new_y2)])
        
        # Plot the original node
        plt.plot(x, y, 'o', color="blue", markersize=10 - _)
        
    # Update the list of nodes
    nodes.extend(new_nodes)

# Display the plot
plt.show()

