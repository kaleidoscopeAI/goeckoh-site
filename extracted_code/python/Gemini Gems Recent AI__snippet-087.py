def calculate_similarity(smiles1, smiles2):
    mol1 = Chem.MolFromSmiles(smiles1)
    mol2 = Chem.MolFromSmiles(smiles2)
    if mol1 and mol2:
        similarity = cube.quantum_inspired_similarity(mol1, mol2)
        return similarity
    return None

