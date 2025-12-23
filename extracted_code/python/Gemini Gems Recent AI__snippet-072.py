def get_molecules():
    molecules = Molecule.query.all()  # Get molecules from database
    #... Serialize molecules to JSON
    return jsonify(molecules)

