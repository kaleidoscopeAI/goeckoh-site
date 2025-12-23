def get_similarity():
    smiles1 = request.json.get('smiles1')
    smiles2 = request.json.get('smiles2')
    task = calculate_similarity.delay(smiles1, smiles2)
    return jsonify({'task_id': task.id})

