import math

def allocate_nodes(data_volume, memory_per_node, min_nodes=1, max_nodes=1000):
    n = math.ceil(data_volume / memory_per_node)
    return max(min_nodes, min(n, max_nodes))
