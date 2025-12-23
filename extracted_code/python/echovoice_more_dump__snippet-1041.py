def __init__(self, stabilizers: Sequence[Sequence[Tuple[int, int]]]) -> None:
    self.stabilizers = list(stabilizers)

def detect(self, state: HybridState) -> List[int]:
    syndromes: List[int] = []
    for S in self.stabilizers:
        xor_sum = 0
        for (node, idx) in S:
            arr = np.asarray(state.E[node]).astype(int).ravel()
            xor_sum ^= int(arr[idx])
        syndromes.append(int(xor_sum))
    return syndromes

def detect_and_repair(self, state: HybridState) -> List[int]:
    synd = self.detect(state)
    return synd

