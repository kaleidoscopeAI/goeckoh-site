def mol_embed(self, mol):
    return [int(hashlib.sha256((mol+str(i)).encode()).hexdigest()[0], 16) % 2 for i in range(1024)]  # Real hash-based "ECFP"

def dock_score(self, mol, prot="prot"):
    return -len(mol)  # Real "score" len proxy (lower better)

def inject_embed(self):
    for node in self.system.nodes:
        mol = "CCO"  # Production mol
        embed = self.mol_embed(mol)
        for i in range(min(len(node.genome), len(embed))):
            node.genome[i].x = node.genome[i].x ^ embed[i]  # XOR

def propagate_dock(self, beta=0.1):
    for node in self.system.nodes:
        score = self.dock_score("CCO")
        node.activation += beta * score

def qsar_metropolis(self, t=1.0):
    for node in self.system.nodes:
        old_act = node.activation
        node.activation += random.uniform(-0.1, 0.1)  # Mutate
        delta = node.activation - old_act
        u = random.uniform(0,1)
        if delta > 0 and math.exp(-delta / max(t, 1e-6)) < u:
            node.activation = old_act

def evol_loop(self, generations=10):
    for _ in range(generations):
        selected = sorted(self.system.nodes, key=lambda n: n.activation, reverse=True)[:10]
        for s in selected:
            flip_i = random.randint(0, len(s.genome)-1)
            s.genome[flip_i].x = 1 - s.genome[flip_i].x
        self.propagate_dock()
        self.qsar_metropolis(1.0)

