def get_features():
    smiles = request.json.get('smiles')
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        features = cube.calculate_features(mol)
        if features is not None:
            scaled_features = cube.scale_features(features)
            return jsonify({'features': scaled_features.tolist()})  # Send scaled features
        else:
            return jsonify({'error': 'Invalid molecule'}), 400
    return jsonify({'error': 'Invalid SMILES'}), 400

