def __init__(self, bonds, temp=1.0):
    self.bonds = bonds
    self.temp = temp

def hamiltonian(self):
    return sum(b.energy() for b in self.bonds)

def local_energy(self, node):
    node_bonds = [b for b in self.bonds if b.node1 is node or b.node2 is node]
    return sum(b.energy() for b in node_bonds) / max(1, len(node_bonds))

