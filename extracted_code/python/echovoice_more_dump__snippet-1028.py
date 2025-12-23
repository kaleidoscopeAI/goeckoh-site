def __init__(self):
    self.values = np.full(5, 0.5)

async def initialize(self):
    self.values = np.random.uniform(0.4, 0.6, 5)

def get_values(self):
    return self.values.tolist()

