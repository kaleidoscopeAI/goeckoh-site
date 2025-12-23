class EndToEndSystem:
    def __init__(self, n_nodes=1024):
        self.nodes = [Node() for _ in range(n_nodes)]
        self.bonds = []
        for _ in range(n_nodes * 2):  # Sparse real adj
            n1, n2 = random.choice(self.nodes), random.choice(self.nodes)
            if n1 is not n2:
                self.bonds.append(Bond(n1, n2))
        self.evolution = EvolutionSystem(self.nodes, self.bonds)
        self.master = MasterUpdate(self)
        self.drug = DrugDiscovery(self)

    def init_bits_nodes(self):
        for node in self.nodes:
            for bit in node.genome:
                bit.x = random.choice([0, 1])  # Real random

    def loop(self, t_max=100):
        for t in range(t_max):
            self.evolution.anneal_step(t)
            self.master.update_state()
            self.master.adaptive_feedback()
            h = EnergyLandscape(self.bonds).hamiltonian()
            print(f"[t={t}] H={h:.2f}")
            checkpoint = self.master.checkpoint()
            print(f"Checkpoint SHA: {checkpoint}")
            if t % 10 == 0:
                self.drug.evol_loop(1)  # Real short evol

