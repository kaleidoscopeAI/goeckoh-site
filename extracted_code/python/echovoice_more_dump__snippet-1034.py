E: Dict[int, np.ndarray]
x: Dict[int, np.ndarray]
E_packed: Dict[int, np.ndarray] = field(default_factory=dict)
metadata: Dict[str, Any] = field(default_factory=dict)

def copy(self) -> "HybridState":
    E_copy = {k: v.copy() for k, v in self.E.items()}
    x_copy = {k: v.copy() for k, v in self.x.items()}
    E_pack_copy = {k: v.copy() for k, v in self.E_packed.items()} if self.E_packed else {}
    return HybridState(E=E_copy, x=x_copy, E_packed=E_pack_copy, metadata=self.metadata.copy())

def pack_all(self) -> None:
    for k, v in self.E.items():
        self.E_packed[k] = pack_bits(v)

