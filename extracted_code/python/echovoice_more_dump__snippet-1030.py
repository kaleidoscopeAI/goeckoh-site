def __init__(self, node_id):
    self.node_id = node_id
    self.position = np.random.rand(3) * 100 - 50
    self.awareness = np.random.rand()
    self.energy = np.random.rand()
    self.valence = np.random.rand()
    self.arousal = np.random.rand()

