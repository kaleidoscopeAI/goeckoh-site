class MolecularCube:
    def calculate_features(self, mol):
        if mol is None:
            return None
        mol_weight = Descriptors.MolWt(mol)
        num_rotatable_bonds = Lipinski.NumRotatableBonds(mol)
        logp = Descriptors.MolLogP(mol)
        return np.array([mol_weight, num_rotatable_bonds, logp])

    def calculate_similarity(self, smiles1, smiles2):
        mol1 = Chem.MolFromSmiles(smiles1)
        mol2 = Chem.MolFromSmiles(smiles2)
        if mol1 is None or mol2 is None:
            return 0.0

        fp1 = AllChem.GetRDKitFP(mol1)  # Use RDKit fingerprints for speed
        fp2 = AllChem.GetRDKitFP(mol2)
        return DataStructs.TanimotoSimilarity(fp1, fp2)

