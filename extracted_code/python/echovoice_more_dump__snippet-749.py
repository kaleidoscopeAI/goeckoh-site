async def get_visualization():
    """Generate real visualization data from quantum states"""
    nodes_data = []
    for node in cognitive_system.cube.nodes[:50]:  # Limit for performance
        nodes_data.append({
            "id": node.node_id,
            "position": node.position.tolist(),
            "awareness": float(node.awareness),
            "energy": float(node.energy),
            "valence": float(node.valence),
            "arousal": float(node.arousal),
            "quantum_alpha": [float(node.quantum_state.alpha.real), 
                            float(node.quantum_state.alpha.imag)]
        })
    
    # Network edges
    edges_data = []
    for edge in list(cognitive_system.cube.graph.edges())[:100]:  # Limit
        edges_data.append({
            "source": edge[0],
            "target": edge[1],
            "strength": cognitive_system.cube.graph[edge[0]][edge[1]].get('weight', 1.0)
        })
    
    return jsonify({
        "nodes": nodes_data,
        "edges": edges_data,
        "quantum_field": cognitive_system.quantum_field.field_strength.tolist(),
        "timestamp": time.time()
    })

