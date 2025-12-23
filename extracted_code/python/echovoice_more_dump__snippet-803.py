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

    def add_bond(self, a: str, b: str, rest_length: float=1.0, stiffness: float =1.0, validated=False):
        key = tuple(sorted([a,b]))
        if key in self.bonds:
            return self.bonds[key]
        bond = Bond(a, b, rest_length, stiffness, validated)
        self.bonds[key] = bond
        self.nodes[a].add_bond(bond)
        self.nodes[b].add_bond(bond)
        return bond

    def compute_all_tensions(self):
        total = 0.0
        for (_, _), bond in self.bonds.items():
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
            moves[nid] = node.pos + move * damping
        for nid, newpos in moves.items():
            self.nodes[nid].pos = newpos

    def detect_crystallization(self, tension_threshold=1e-3, stable_steps=10):
        cur_t = self.compute_all_tensions()
        now = time.time()
        self.crystal_states.append((now, cur_t))
        window = [t for ts, t in self.crystal_states[-stable_steps:]]
        if len(window) < stable_steps:
            return False
        if max(window) - min(window) < tension_threshold:
            self.last_crystallization = {"time": now, "tension": cur_t, "nodes": {nid: n.pos.tolist() for nid, n in self.nodes.items()}}
            return True
        return False

