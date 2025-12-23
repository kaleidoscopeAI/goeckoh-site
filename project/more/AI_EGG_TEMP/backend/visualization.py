# backend/visualization.py
import logging

def format_state(simulator):
    """
    Prepare the current simulation state as a dictionary for JSON transmission.
    Includes node positions, bonds, and metrics.
    """
    node_stress = {nid: 0.0 for nid in simulator.nodes.keys()}
    for (a, b), bond in simulator.bonds.items():
        if bond.broken:
            continue
        s = getattr(bond, 'last_stress', 0.0)
        if s > node_stress.get(a, 0.0):
            node_stress[a] = s
        if s > node_stress.get(b, 0.0):
            node_stress[b] = s

    nodes_data = []
    for nid, node in simulator.nodes.items():
        x, y, z = node.pos
        stress_val = node_stress.get(nid, 0.0)
        nodes_data.append([nid, round(x, 4), round(y, 4), round(z, 4), round(stress_val, 4)])
    nodes_data.sort(key=lambda entry: entry[0])

    bonds_data = []
    for (a, b), bond in simulator.bonds.items():
        if bond.broken:
            continue
        bonds_data.append([a, b])

    metrics = {
        "stress": round(simulator.global_stress, 4),
        "harmony": round(simulator.harmony, 4),
        "emergence": round(simulator.emergence, 4),
        "confidence": round(simulator.confidence, 4)
    }
    return {"nodes": nodes_data, "bonds": bonds_data, "metrics": metrics}
