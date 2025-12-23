def __init__(self):
    self.graph = nx.Graph()
    self.features = {}

def build_from_smiles(self, smiles: str) -> None:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Invalid SMILES string")

    for atom in mol.GetAtoms():
        self.graph.add_node(atom.GetIdx(), 
                          atomic_num=atom.GetAtomicNum(),
                          formal_charge=atom.GetFormalCharge(),
                          hybridization=atom.GetHybridization(),
                          is_aromatic=atom.GetIsAromatic())

    for bond in mol.GetBonds():
        self.graph.add_edge(bond.GetBeginAtomIdx(),
                          bond.GetEndAtomIdx(),
                          bond_type=bond.GetBondType())

def calculate_molecular_descriptors(self) -> Dict:
    descriptors = {
        'num_atoms': self.graph.number_of_nodes(),
        'num_bonds': self.graph.number_of_edges(),
        'molecular_weight': sum(data['atomic_num'] for _, data in self.graph.nodes(data=True)),
        'topological_index': nx.wiener_index(self.graph)
    }
    return descriptors

