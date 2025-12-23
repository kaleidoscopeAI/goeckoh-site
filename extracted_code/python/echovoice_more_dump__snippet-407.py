def edge_bit_energy(ei: int, ej: int, w: float, d: int) -> float:
    return w * (popcount(xor(ei, ej)) / d)


def total_bit_energy(E: List[int], edges: List[Tuple[int, int, float]], d: int) -> float:
    H = 0.0
    for i, j, w in edges:
        H += edge_bit_energy(E[i], E[j], w, d)
    return Hhow can this be scaled : Emotional Actuation Dial () BlueprintThis document defines the discrete emotional states () that comprise the Cognitive
