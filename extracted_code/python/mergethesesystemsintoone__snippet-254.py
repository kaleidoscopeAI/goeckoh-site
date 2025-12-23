def sample(self, network: BayesianNetwork, num_samples: int) -> List[Dict]:
    """
    Performs Markov Chain Monte Carlo sampling on the Bayesian Network.
    """
    samples = []
    current_state = self._initialize_state(network)

    for i in range(num_samples):
        for node_id in network.nodes:
            new_state = self._propose_state(current_state, node_id)
            acceptance_prob = self._calculate_acceptance_probability(current_state, new_state, network)
            if random.random() < acceptance_prob:
                current_state = new_state
        if i > self.burn_in and (i - self.burn_in) % self.sample_interval == 0:
            samples.append(current_state.copy())
    return samples

def _initialize_state(self, network: BayesianNetwork) -> Dict:
  """Initializes a random state for the Markov Chain."""
  initial_state = {}
  for node_id, data in network.nodes.items():
    initial_state [node_id] = random.random()
  return initial_state

def _propose_state(self, current_state: Dict, node_id: str) -> Dict:
  """Proposes a new state for a given node."""
  new_state = current_state.copy()
  new_state [node_id] = random.random()
  return new_state

def _calculate_acceptance_probability(self, current_state: Dict, new_state: Dict, network: BayesianNetwork) -> float:
    """Calculates the Metropolis acceptance probability."""
    current_prob = self._calculate_state_probability(current_state, network)
    new_prob = self._calculate_state_probability(new_state, network)
    return min(1, new_prob/current_prob) if current_prob > 0 else 1.0

def _calculate_state_probability(self, state: Dict, network: BayesianNetwork) -> float:
    """Calculates the probability of a given state."""
    total_prob = 1.0
    for node_id, value in state.items():
        node_prob = network.nodes[node_id].get('strength', 0.0)
        total_prob *= (value * node_prob + (1 - value) * (1 - node_prob))
    return total_prob
