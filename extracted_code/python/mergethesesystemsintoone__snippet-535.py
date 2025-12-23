def __init__(self) -> None:
    self.phi_history: List[float] = []

def entropy(self, data: List[float]) -> float:
    tensor = torch.tensor(data, dtype=torch.float32)
    probs = torch.softmax(tensor, dim=0)
    return -torch.sum(probs * torch.log(probs + 1e-10)).item()

def integrated_information(self, vec: List[float]) -> float:
    n = len(vec)
    parts = max(1, n // 2)
    sys_entropy = self.entropy(vec)
    part_entropy = sum(self.entropy(vec[i::parts]) for i in range(parts))
    phi_val = max(0.0, sys_entropy - part_entropy)
    self.phi_history.append(phi_val)
    return phi_val

