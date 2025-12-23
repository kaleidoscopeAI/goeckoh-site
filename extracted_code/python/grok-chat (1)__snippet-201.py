def __init__(self, n_nodes=1024):
    self.nodes = [Node() for _ in range(n_nodes)]
    self.bonds = [Bond(random.choice(self.nodes), random.choice(self.nodes)) for _ in range(n_nodes * 2)]  # Sparse adj
    self.evolution = EvolutionSystem(self.nodes, self.bonds)
    self.master = MasterUpdate(self)
    self.drug = DrugDiscovery(self)

def init_bits_nodes(self):
    print("Initialized")

def loop(self, t_max=100):
    for t in range(t_max):
        self.evolution.anneal_step(t)
        self.master.update_state()
        self.master.adaptive_feedback()
        h = EnergyLandscape(self.bonds).hamiltonian()
        print(f"[t={t}] H={h:.2f}")
        try:
            checkpoint = self.master.checkpoint()
            print(f"Checkpoint SHA: {checkpoint}")
        except Exception as e:
            print(f"Error: {e}")

