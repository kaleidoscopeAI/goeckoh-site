class Molecule(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    smiles = db.Column(db.String, unique=True, nullable=False)
    mol_weight = db.Column(db.Float)
    num_rotatable_bonds = db.Column(db.Integer)
    logp = db.Column(db.Float)
    #... other features

    def __init__(self, smiles):
        self.smiles = smiles
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            features = cube.calculate_features(mol)
            self.mol_weight = features
            self.num_rotatable_bonds = int(features) #Cast to int
            self.logp = features
            #... other features

