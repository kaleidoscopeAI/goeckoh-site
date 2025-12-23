def simple_edges(N: int, k: int = 6) -> np.ndarray:
    edges = []
    for i in range(N):
        edges.append((i, (i + 1) % N))
        for j in range(1, k // 2 + 1):
            edges.append((i, (i + j) % N))
    if not edges: return np.zeros((0,2), dtype=np.int32)
    return np.array(sorted({tuple(sorted(e)) for e in edges}), dtype=np.int32)

