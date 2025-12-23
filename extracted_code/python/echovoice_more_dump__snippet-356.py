class EvolutionaryAgent:
    def __init__(self, dimensions, bounds=None, mutation_rate=0.1):
        self.dimensions = dimensions
        self.bounds = bounds if bounds else [(-10, 10) for _ in range(dimensions)]
        self.mutation_rate = mutation_rate
        
        # Initialize position and velocity
        self.position = np.array([np.random.uniform(low, high) for low, high in self.bounds])
        self.velocity = np.random.uniform(-1, 1, dimensions)
        
        # Best known position and fitness
        self.best_position = self.position.copy()
        self.best_fitness = float('-inf')
        
        # Genetic properties for evolution
        self.dna = np.random.uniform(0, 1, dimensions * 2)  # Genes influence behavior
        
        # Adaptation parameters
        self.cognitive = 1.5 + 0.5 * self.dna[0]  # Personal best influence
        self.social = 1.5 + 0.5 * self.dna[1]     # Global best influence
        self.inertia = 0.5 + 0.4 * self.dna[2]    # Velocity retention
        self.exploration = 0.1 + 0.9 * self.dna[3]  # Randomness in movement
        
    def update(self, global_best, chaos_field=None):
        """Update agent position using swarm intelligence and chaos influence"""
        # Standard PSO update with evolutionary parameters
        r1, r2 = np.random.random(2)
        cognitive_velocity = self.cognitive * r1 * (self.best_position - self.position)
        social_velocity = self.social * r2 * (global_best - self.position)
        
        # Apply chaos field influence if provided
        chaos_velocity = np.zeros(self.dimensions)
        if chaos_field is not None:
            # Sample from chaos field distributions
            chaos_sample = np.random.choice(len(chaos_field), self.dimensions, p=chaos_field)
            chaos_direction = chaos_sample / (len(chaos_field) - 1) * 2 - 1  # -1 to 1
            chaos_velocity = self.exploration * chaos_direction
        
        # Update velocity with inertia
        self.velocity = (self.inertia * self.velocity + 
                         cognitive_velocity + 
                         social_velocity + 
                         chaos_velocity)
        
        # Apply velocity constraints
        max_velocity = 0.1 * np.array([high - low for low, high in self.bounds])
        self.velocity = np.clip(self.velocity, -max_velocity, max_velocity)
        
        # Update position
        self.position += self.velocity
        
        # Apply boundary constraints
        for i in range(self.dimensions):
            if self.position[i] < self.bounds[i][0]:
                self.position[i] = self.bounds[i][0]
                self.velocity[i] *= -0.5  # Bounce with damping
            elif self.position[i] > self.bounds[i][1]:
                self.position[i] = self.bounds[i][1]
                self.velocity[i] *= -0.5  # Bounce with damping
        
        return self.position
    
    def evolve(self, fitness, other_agent):
        """Evolve the agent through genetic operations with another agent"""
        if fitness > self.best_fitness:
            self.best_fitness = fitness
            self.best_position = self.position.copy()
            
        # Only evolve if this agent is less fit than the other
        if self.best_fitness < other_agent.best_fitness:
            # Crossover
            crossover_point = np.random.randint(1, len(self.dna) - 1)
            new_dna = np.concatenate([
                self.dna[:crossover_point],
                other_agent.dna[crossover_point:]
            ])
            
            # Mutation
            mutation_mask = np.random.random(len(new_dna)) < self.mutation_rate
            mutation_values = np.random.uniform(-0.2, 0.2, len(new_dna))
            new_dna[mutation_mask] += mutation_values[mutation_mask]
            new_dna = np.clip(new_dna, 0, 1)
            
            # Update DNA and derived parameters
            self.dna = new_dna
            self.cognitive = 1.5 + 0.5 * self.dna[0]
            self.social = 1.5 + 0.5 * self.dna[1]
            self.inertia = 0.5 + 0.4 * self.dna[2]
            self.exploration = 0.1 + 0.9 * self.dna[3]
            
            # Slight movement toward better agent's position
            self.position += 0.1 * (other_agent.best_position - self.position)


