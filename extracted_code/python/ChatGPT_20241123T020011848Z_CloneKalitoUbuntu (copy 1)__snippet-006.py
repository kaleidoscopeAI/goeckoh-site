import matplotlib.pyplot as plt
import random
import time

# Set up figure for live updating
plt.ion()  # Turn on interactive mode
fig, ax = plt.subplots(figsize=(8, 8))
ax.set_title("Seed-Based Node Growth Simulation (Cycle-by-Cycle)")
ax.set_xlim(-10, 10)
ax.set_ylim(-10, 10)

# Starting position of the seed node
nodes = [(0, 0)]  # Seed node at the center

# Parameters
replications = 8  # Number of replication rounds
offset_range = 1  # Distance from parent node for each replication
colors = ["blue", "purple", "cyan", "green", "orange", "red"]

# Simulation of node replication with adaptation over time
for cycle in range(replications):
    new_nodes = []
    
    for x, y in nodes:
        # Generate two new nodes for each existing node with slight random offsets
        new_x1, new_y1 = x + random.uniform(-offset_range, offset_range), y + random.uniform(-offset_range, offset_range)
        new_x2, new_y2 = x + random.uniform(-offset_range, offset_range), y + random.uniform(-offset_range, offset_range)
        
        # Append new nodes to the list
        new_nodes.extend([(new_x1, new_y1), (new_x2, new_y2)])
        
        # Plot the original node
        ax.plot(x, y, 'o', color=colors[cycle % len(colors)], markersize=8 - cycle * 0.5)
    
    # Update nodes list with newly generated nodes
    nodes.extend(new_nodes)
    
    # Pause to visualize each replication cycle
    plt.draw()
    plt.pause(1)  # 1-second delay between cycles to simulate growth over time

plt.ioff()  # Turn off interactive mode
plt.show()

