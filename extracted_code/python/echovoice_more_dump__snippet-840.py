class UnifiedInterfaceNode:
    id: str
    valence: float = 0.0   # [-1,1]
    arousal: float = 0.0   # [0,1]
    stance: float = 0.0
    coherence: float = 1.0
    energy: float = 1.0    # [0,1]
    knowledge: float = 0.0 # [0,1]
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

# Bond and Node (from node_graph.py)
class Bond:
    def __init__(self, a: str, b: str, rest_length: float = 1.0, stiffness: float = 1.0, validated: bool = False):
        self.a = a
        self.b = b
        self.rest_length = rest_length
        self.stiffness = stiffness
        self.validated = validated
        self.tension = 0.0

    def compute_tension(self, pos_a: np.ndarray, pos_b: np.ndarray):
        dist = float(np.linalg.norm(pos_a - pos_b))
        self.tension = self.stiffness * max(0.0, abs(dist - self.rest_length))
        return self.tension

class Node:
    def __init__(self, node_id: str, pos: np.ndarray = None, fixed=False):
        self.id = node_id
        self.pos = np.array(pos) if pos is not None else np.random.randn(3).astype(float)
        self.fixed = fixed
        self.energy = 1.0
        self.valence = 0.0
        self.arousal = 0.0
        self.bonds: List[Bond] = []

    def add_bond(self, b: Bond):
        self.bonds.append(b)

class NodeGraph:
    def __init__(self):
        self.nodes: Dict[str, Node] = {}
        self.bonds: Dict[Tuple[str,str], Bond] = {}
        self.last_crystallization = None
        self.crystal_states = []

    def add_node(self, node_id: str, pos=None, fixed=False):
        if node_id in self.nodes:
            return self.nodes[node_id]
        n = Node(node_id, pos=pos, fixed=fixed)
        self.nodes[node_id] = n
        return n

    def add_bond(self, a: str, b: str, rest_length: float=1.0, stiffness: float=1.0, validated=False):
        key = tuple(sorted([a,b]))
        if key in self.bonds:
            return self.bonds[key]
        bond = Bond(a,b,rest_length,stiffness,validated)
        self.bonds[key] = bond
        self.nodes[a].add_bond(bond)
        self.nodes[b].add_bond(bond)
        return bond

    def compute_all_tensions(self):
        total = 0.0
        for (a,b), bond in self.bonds.items():
            pos_a = self.nodes[bond.a].pos
            pos_b = self.nodes[bond.b].pos
            total += bond.compute_tension(pos_a, pos_b)
        return total

    def relax_step(self, step_size=0.01, damping=0.9):
        moves = {}
        for nid, node in self.nodes.items():
            if node.fixed:
                continue
            net_force = np.zeros_like(node.pos, dtype=float)
            for bond in node.bonds:
                other = bond.a if bond.b == nid else bond.b
                other_pos = self.nodes[other].pos
                vec = node.pos - other_pos
                dist = np.linalg.norm(vec) + 1e-8
                direction = vec / dist
                fmag = -bond.stiffness * (dist - bond.rest_length)
                net_force += fmag * direction
            move = step_size * net_force
            moves[nid] = (node.pos + move * damping)
        for nid, newpos in moves.items():
            self.nodes[nid].pos = newpos

    def detect_crystallization(self, tension_threshold=1e-3, stable_steps=10):
        cur_t = self.compute_all_tensions()
        now = time.time()
        self.crystal_states.append((now, cur_t))
        window = [t for ts,t in self.crystal_states[-stable_steps:]]
        if len(window) < stable_steps:
            return False
        if max(window) - min(window) < tension_threshold:
            self.last_crystallization = {"time": now, "tension": cur_t, "nodes": {nid: n.pos.copy().tolist() for nid,n in self.nodes.items()}}
            return True
        return False

# QSINEnvelope (from qsintensor.py)
class QSINEnvelope:
    def __init__(self, d0=64, d1=64, d2=16, d3=8):
        self.tensor = np.zeros((d0,d1,d2,d3), dtype=np.float32)

    def update(self, idxs, value):
        d0,d1,d2,d3 = idxs
        self.tensor[d0,d1,d2,d3] = value

    def marginalize(self, axis):
        return self.tensor.sum(axis=axis)

