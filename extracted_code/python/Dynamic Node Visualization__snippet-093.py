class UnifiedInterfaceNode:
    id: str
    valence: float = 0.0
    arousal: float = 0.0
    stance: float = 0.0
    coherence: float = 1.0
    energy: float = 1.0
    knowledge: float = 0.0
    hamiltonian_e: float = 0.0
    perspective_v: np.ndarray = field(default_factory=lambda: np.zeros(64))
    governance_flags: Dict[str, Any] = field(default_factory=lambda: {"L0":True, "L1":True})
    history: List[Dict[str,Any]] = field(default_factory=list)
    last_update: float = field(default_factory=time.time)

    def snapshot(self):
        return {
            "id": self.id,
            "valence": self.valence,
            "arousal": self.arousal,
            "energy": self.energy,
            "hamiltonian_e": self.hamiltonian_e,
            "time": time.time()
        }

    def apply_feedback(self, delta_valence=0.0, delta_arousal=0.0, delta_energy=0.0):
        self.valence = float(np.clip(self.valence + delta_valence, -1.0, 1.0))
        self.arousal = float(np.clip(self.arousal + delta_arousal, 0.0, 1.0))
        self.energy = float(np.clip(self.energy + delta_energy, 0.0, 1.0))
        self.last_update = time.time()
        self.history.append(self.snapshot())

    def update_perspective(self, vec: np.ndarray, alpha=0.2):
        if self.perspective_v.shape != vec.shape:
            self.perspective_v = np.zeros_like(vec)
        self.perspective_v = (1-alpha)*self.perspective_v + alpha*vec

