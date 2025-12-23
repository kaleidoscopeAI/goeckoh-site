def calculate_phi(emb: np.ndarray) -> float:
    if emb.size == 0:
        return 0.0
    entropy = lambda data: -np.sum(data * np.log(data + 1e-12)) if np.sum(data) > 0 else 0
    data = np.abs(emb) / (np.sum(np.abs(emb)) + 1e-12)
    sys_entropy = entropy(data)
    parts = 2
    part_entropy = 0
    for i in range(parts):
        partition = data[i::parts]
        part_entropy += entropy(partition) / parts
    return max(0, sys_entropy - part_entropy)

