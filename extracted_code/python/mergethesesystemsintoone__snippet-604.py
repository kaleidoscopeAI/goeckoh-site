def __init__(self, burn_in: int = 100, sample_interval: int = 10):
    self.burn_in = burn_in
    self.sample_interval = sample_interval

def sample(self, network: BayesianNetwork, num_samples: int) -> List[Dict]:
    """
    Performs Markov Chain Monte Carlo sampling on the Bayesian Network.
    """
    samples =
