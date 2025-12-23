def __init__(self, fitness_func, dimensions, swarm_size=50, 
             bounds=None, use_chaos=True, chaos_size=100, chaos_rule=110):
    self.fitness_func = fitness_func
    self.dimensions = dimensions
    self.swarm_size = swarm_size
    self.bounds = bounds if bounds else [(-10, 10) for _ in range(dimensions)]

    # Initialize swarm
    self.agents = [EvolutionaryAgent(dimensions, bounds) for _ in range(swarm_size)]

    # Global best
    self.global_best_position = None
    self.global_best_fitness = float('-inf')

    # Chaos field generator
    self.use_chaos = use_chaos
    if use_chaos:
        self.chaos_generator = CellularChaosGenerator(size=chaos_size, rule=chaos_rule)
        self.chaos_field = None
        self.update_chaos_field()

    # Statistics and properties
    self.convergence_history = []
    self.diversity_history = []
    self.iteration = 0

def update_chaos_field(self):
    """Update the chaos field by evolving cellular automata"""
    if self.use_chaos:
        # Evolve the cellular automaton
        self.chaos_generator.evolve(steps=5)

        # Extract probability distributions for swarm guidance
        entropy_x, entropy_y, edge_features = self.chaos_generator.get_chaos_features()

        # Store as flattened probability distribution
        self.chaos_field = edge_features

def evaluate_fitness(self, positions):
    """Evaluate fitness for multiple positions in parallel"""
    # Use multiprocessing for fitness evaluation
    with Pool(min(cpu_count(), len(positions))) as pool:
        return pool.map(self.fitness_func, positions)

def optimize(self, iterations=100):
    """Run the optimization process"""
    for iteration in range(iterations):
        self.iteration = iteration

        # Update chaos field
        if self.use_chaos and iteration % 5 == 0:
            self.update_chaos_field()

        # Get all agent positions
        positions = [agent.position for agent in self.agents]

        # Evaluate fitness in parallel
        fitness_values = self.evaluate_fitness(positions)

        # Update agent personal bests and global best
        for i, (position, fitness) in enumerate(zip(positions, fitness_values)):
            agent = self.agents[i]

            if fitness > agent.best_fitness:
                agent.best_fitness = fitness
                agent.best_position = position.copy()

            if fitness > self.global_best_fitness:
                self.global_best_fitness = fitness
                self.global_best_position = position.copy()

        # Update agents' positions
        for agent in self.agents:
            agent.update(self.global_best_position, self.chaos_field)

        # Evolutionary step - arrange agents in random pairs and evolve
        indices = list(range(self.swarm_size))
        np.random.shuffle(indices)

        for i in range(0, self.swarm_size - 1, 2):
            idx1, idx2 = indices[i], indices[i+1]
            agent1, agent2 = self.agents[idx1], self.agents[idx2]

            # Cross-evolve
            agent1.evolve(fitness_values[idx1], agent2)
            agent2.evolve(fitness_values[idx2], agent1)

        # Calculate diversity
        positions = np.array([agent.position for agent in self.agents])
        diversity = np.mean(np.std(positions, axis=0))

        # Record history
        self.convergence_history.append(self.global_best_fitness)
        self.diversity_history.append(diversity)

        # Dynamic parameter adjustment based on diversity
        if iteration > 10 and diversity < 0.01:
            # Inject chaos to escape local optima
            for i in range(self.swarm_size // 10):  # Reset 10% of agents
                idx = np.random.randint(0, self.swarm_size)
                self.agents[idx] = EvolutionaryAgent(self.dimensions, self.bounds)

        # Progress report every 10 iterations
        if iteration % 10 == 0:
            print(f"Iteration {iteration}: Best fitness = {self.global_best_fitness:.4f}, Diversity = {diversity:.4f}")

    return self.global_best_position, self.global_best_fitness

def visualize_optimization(self):
    """Visualize the optimization process"""
    plt.figure(figsize=(15, 10))

    # Plot convergence
    plt.subplot(2, 2, 1)
    plt.plot(self.convergence_history)
    plt.title('Convergence History')
    plt.xlabel('Iteration')
    plt.ylabel('Best Fitness')

    # Plot diversity
    plt.subplot(2, 2, 2)
    plt.plot(self.diversity_history)
    plt.title('Swarm Diversity')
    plt.xlabel('Iteration')
    plt.ylabel('Diversity')

    # Plot final agent positions (first 2 dimensions)
    if self.dimensions >= 2:
        plt.subplot(2, 2, 3)
        positions = np.array([agent.position for agent in self.agents])
        plt.scatter(positions[:, 0], positions[:, 1], alpha=0.6)
        plt.scatter([self.global_best_position[0]], [self.global_best_position[1]], 
                   color='red', s=100, marker='*')
        plt.title('Agent Positions (dims 0-1)')
        plt.xlabel('Dimension 0')
        plt.ylabel('Dimension 1')

    # Plot chaos field if used
    if self.use_chaos:
        plt.subplot(2, 2, 4)
        plt.imshow(self.chaos_generator.grid, cmap='binary')
        plt.title('Chaos Field')
        plt.colorbar()

    plt.tight_layout()
    plt.show()

def visualize_search_space(self, resolution=50):
    """Visualize the fitness landscape for 2D problems"""
    if self.dimensions != 2:
        print("Visualization only available for 2D problems")
        return

    # Create grid for visualization
    x = np.linspace(self.bounds[0][0], self.bounds[0][1], resolution)
    y = np.linspace(self.bounds[1][0], self.bounds[1][1], resolution)
    X, Y = np.meshgrid(x, y)

    # Evaluate fitness across the grid
    Z = np.zeros((resolution, resolution))
    for i in range(resolution):
        positions = []
        for j in range(resolution):
            positions.append(np.array([X[i, j], Y[i, j]]))

        Z[i, :] = self.evaluate_fitness(positions)

    # Plot the fitness landscape
    plt.figure(figsize=(12, 10))

    # Contour plot
    plt.contourf(X, Y, Z, 50, cmap='viridis', alpha=0.5)
    plt.colorbar(label='Fitness')

    # Plot agent positions
    positions = np.array([agent.position for agent in self.agents])
    plt.scatter(positions[:, 0], positions[:, 1], color='white', alpha=0.6, label='Agents')

    # Plot global best
    plt.scatter([self.global_best_position[0]], [self.global_best_position[1]], 
               color='red', s=200, marker='*', label='Global Best')

    plt.title('Fitness Landscape and Agent Positions')
    plt.xlabel('Dimension 0')
    plt.ylabel('Dimension 1')
    plt.legend()
    plt.tight_layout()
    plt.show()


