def __init__(self, size=100, rule=110):
    self.size = size
    self.rule = rule
    self.grid = np.zeros((size, size), dtype=np.int8)
    self.initialize_grid()

def initialize_grid(self):
    """Initialize with a single cell or random pattern"""
    # Middle cell initialization
    self.grid[0, self.size // 2] = 1

def rule_to_transitions(self):
    """Convert rule number to transition dictionary"""
    transitions = {}
    rule_binary = format(self.rule, '08b')
    patterns = ['111', '110', '101', '100', '011', '010', '001', '000']
    for i, pattern in enumerate(patterns):
        transitions[pattern] = int(rule_binary[i])
    return transitions

def evolve(self, steps=1):
    """Evolve the cellular automaton for specified steps"""
    transitions = self.rule_to_transitions()

    for _ in range(steps):
        new_grid = np.zeros_like(self.grid)

        for i in range(self.size):
            # Get the pattern for each cell including wrapping
            left = np.roll(self.grid[i], 1)
            right = np.roll(self.grid[i], -1)

            # Combine to get neighborhood patterns
            patterns = np.vstack((left, self.grid[i], right)).T

            # Apply rules
            for j in range(self.size):
                pattern = ''.join(map(str, patterns[j]))
                new_grid[i, j] = transitions.get(pattern, 0)

        self.grid = new_grid

    return self.grid

def get_chaos_features(self):
    """Extract features from the chaos pattern for swarm guidance"""
    # Calculate entropy along rows and columns
    entropy_x = np.zeros(self.size)
    entropy_y = np.zeros(self.size)

    for i in range(self.size):
        # Calculate row and column distributions
        row_vals, row_counts = np.unique(self.grid[i], return_counts=True)
        col_vals, col_counts = np.unique(self.grid[:, i], return_counts=True)

        # Calculate entropy (if non-zero distributions)
        if len(row_counts) > 1:
            row_probs = row_counts / np.sum(row_counts)
            entropy_x[i] = -np.sum(row_probs * np.log2(row_probs))

        if len(col_counts) > 1:
            col_probs = col_counts / np.sum(col_counts)
            entropy_y[i] = -np.sum(col_probs * np.log2(col_probs))

    # Detect structures using convolution
    edge_kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
    edge_response = convolve2d(self.grid, edge_kernel, mode='same', boundary='wrap')

    # Return features as probability distributions
    entropy_x = entropy_x / np.sum(entropy_x) if np.sum(entropy_x) > 0 else np.ones(self.size) / self.size
    entropy_y = entropy_y / np.sum(entropy_y) if np.sum(entropy_y) > 0 else np.ones(self.size) / self.size
    edge_features = np.abs(edge_response.flatten())
    edge_features = edge_features / np.sum(edge_features) if np.sum(edge_features) > 0 else np.ones(self.size*self.size) / (self.size*self.size)

    return entropy_x, entropy_y, edge_features


