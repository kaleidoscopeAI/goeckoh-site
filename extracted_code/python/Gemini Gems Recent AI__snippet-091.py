def get_similarity():
    smiles1 = request.json.get('smiles1')
    smiles2 = request.json.get('smiles2')
    similarity = cube.calculate_similarity(smiles1, smiles2)
    return jsonify({'similarity': similarity})

