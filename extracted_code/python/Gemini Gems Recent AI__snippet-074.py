def get_similar_molecules(smiles):
    similar = cube.find_similar_molecules(smiles)
    return jsonify(similar)  # Return data as JSON

