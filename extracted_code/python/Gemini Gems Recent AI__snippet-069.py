def find_similar_classical(n_clicks, smiles):
    if n_clicks > 0 and smiles:
        similar = cube.find_similar_molecules(smiles)
        if similar:
            results_list = html.Ul([html.Li(f"{smi} (Similarity: {sim:.2f})") for smi, sim in similar])
        else:
            results_list = "No similar molecules found."
        return results_list
    return ""

