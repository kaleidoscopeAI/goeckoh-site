for node in nodes:
    if "energy" in node.resources:
        node.resources["energy"] -= 0.1
        if node.resources["energy"] < 0.1:
            node.logs.append("Energy critically low. Seeking shared resources.")
            # Example resource sharing logic
            for other_node in nodes:
                if other_node.node_id != node.node_id and "energy" in other_node.resources:
                    shared_energy = other_node.resources["energy"] * 0.1
                    node.resources["energy"] += shared_energy
                    other_node.resources["energy"] -= shared_energy
                    node.logs.append(f"Received {shared_energy:.2f} energy from Node {other_node.node_id}.")

