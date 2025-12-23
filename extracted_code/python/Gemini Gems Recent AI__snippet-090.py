def get_molecules():
    molecules = Molecule.query.all()
    molecule_list =
    for molecule in molecules:
        molecule_list.append({
            'id': molecule.id,
            'smiles': molecule.smiles,
            'mol_weight': molecule.mol_weight,
            'num_rotatable_bonds': molecule.num_rotatable_bonds,
            'logp': molecule.logp,
            #... other features
        })
    return jsonify(molecule_list)

