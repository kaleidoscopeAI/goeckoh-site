#Reconstruct the graph from JSON (if needed) - For large datasets, this might be a bottleneck
# cube.graph.clear()
# cube.graph.add_nodes_from(cube_data.keys())
# for node, data in cube_data.items():
#     cube.graph.nodes[node].update(data)
cube.visualize_cube() #Just call the existing visualization method
return cube.visualize_cube()

