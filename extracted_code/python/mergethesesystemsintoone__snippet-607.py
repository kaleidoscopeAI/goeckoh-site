def generate_molecule(node, output_file):
    """Generate a 3D molecule from cube node data and save it as PDB."""
    smiles = "C" * int(node["energy"] * 2)  # Simplified molecular representation
    molecule = Chem.MolFromSmiles(smiles)
    molecule = AllChem.AddHs(molecule)
    AllChem.EmbedMolecule(molecule)
    AllChem.UFFOptimizeMolecule(molecule)
    with open(output_file, "w") as f:
        f.write(Chem.MolToPDBBlock(molecule))
    print(f"Molecule saved to {output_file}")
    return molecule

