def create_new_node(existing_nodes, dna_base):
    node_id_counter = len(existing_nodes)
    new_node = Node(node_id_counter, f"{dna_base}_child{node_id_counter}", resources={})
    existing_nodes.append(new_node)

