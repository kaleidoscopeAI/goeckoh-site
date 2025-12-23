def __init__(self, n_nodes=64, input_dim=8):
    self.nodes = [OrganicNode(i, data_vector=np.random.rand(input_dim)) for i in range(n_nodes)]
    self.transformer = BitLevelTransformer(input_dim)
    # ...

def reflect_supernodes(self, supernodes):
    for sn in supernodes:
        input_tensor = torch.tensor(np.stack([node.vector for node in sn.nodes]), dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            output = self.transformer(input_tensor)
        projected = output.squeeze(0).numpy()
        for i, node in enumerate(sn.nodes):
            node.vector = projected[i]
