def __init__(self, nodes):
    self.nodes = nodes
    self.prototype = np.mean([n.vector for n in nodes], axis=0)
    self.energy = np.mean([n.energy for n in nodes])

def reflect(self, transformer):
    input_tensor = torch.tensor(self.prototype, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        output = transformer(input_tensor)
    self.prototype = output.squeeze().numpy()

