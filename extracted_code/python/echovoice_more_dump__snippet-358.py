class SwarmGraphOptimizer:
    def __init__(self, graph, dimensions=3, swarm_size=100, use_chaos=True):
        self.graph = graph
        self.dimensions = dimensions
        self.swarm_size = swarm_size
        self.use_chaos = use_chaos
        
        # Extract graph properties for optimization
        self.node_positions = {}
        self.edge_weights = {}
        self.node_properties = {}
        
        # Initialize from graph
        self._extract_graph_properties()
        
        # Setup the optimizer
        self.optimizer = ChaosSwarmOptimizer(
            fitness_func=self._graph_fitness_function,
            dimensions=dimensions * len(self.graph.nodes),
            swarm_size=swarm_size,
            bounds=[(-5, 5) for _ in range(dimensions * len(self.graph.nodes))],
            use_chaos=use_chaos
        )
    
    def _extract_graph_properties(self):
        """Extract properties from the graph for optimization"""
        for node in self.graph.nodes:
            # Get or generate node positions
            if 'position' in self.graph.nodes[node]:
                self.node_positions[node] = self.graph.nodes[node]['position']
            else:
                self.node_positions[node] = np.random.normal(0, 1, self.dimensions)
            
            # Extract other node properties
            props = {}
            for key, value in self.graph.nodes[node].items():
                if key != 'position' and isinstance(value, (int, float)):
                    props[key] = value
            self.node_properties[node] = props
        
        # Extract edge weights
        for u, v in self.graph.edges:
            self.edge_weights[(u, v)] = self.graph.edges[u, v].get('weight', 1.0)
    
    def _graph_fitness_function(self, position_vector):
        """Fitness function for graph optimization"""
        # Reshape vector to node positions
        node_positions = {}
        n_nodes = len(self.graph.nodes)
        position_matrix = position_vector.reshape(n_nodes, self.dimensions)
        
        for i, node in enumerate(self.graph.nodes):
            node_positions[node] = position_matrix[i]
        
        # Calculate fitness based on several factors
        
        # 1. Edge length optimization - shorter is better
        edge_length_factor = 0
        for u, v in self.graph.edges:
            distance = np.linalg.norm(node_positions[u] - node_positions[v])
            weight = self.edge_weights.get((u, v), 1.0)
            edge_length_factor += weight * distance
        
        # 2. Node distribution - more evenly distributed is better
        distribution_factor = 0
        for i, u in enumerate(self.graph.nodes):
            for v in self.graph.nodes:
                if u != v:
                    distance = np.linalg.norm(node_positions[u] - node_positions[v])
                    distribution_factor += 1 / (distance + 0.1)  # Avoid division by zero
        
        # 3. Property-based relationships - nodes with similar properties should be closer
        property_factor = 0
        for u in self.graph.nodes:
            for v in self.graph.nodes:
                if u != v:
                    # Calculate property similarity
                    similarity = 0
                    count = 0
                    for key in set(self.node_properties[u].keys()) & set(self.node_properties[v].keys()):
                        diff = abs(self.node_properties[u][key] - self.node_properties[v][key])
                        max_val = max(abs(self.node_properties[u][key]), abs(self.node_properties[v][key]))
                        if max_val > 0:
                            similarity += 1 - (diff / max_val)
                            count += 1
                    
                    if count > 0:
                        similarity /= count
                        distance = np.linalg.norm(node_positions[u] - node_positions[v])
                        # Similar nodes should be closer
                        property_factor += similarity / (distance + 0.1)
        
        # Combine factors with appropriate weights
        fitness = -edge_length_factor - 0.1 * distribution_factor + 0.5 * property_factor
        
        return fitness
    
    def optimize_graph_layout(self, iterations=100):
        """Optimize the graph layout using swarm intelligence"""
        # Run the optimizer
        best_position, best_fitness = self.optimizer.optimize(iterations=iterations)
        
        # Update graph with optimized positions
        n_nodes = len(self.graph.nodes)
        position_matrix = best_position.reshape(n_nodes, self.dimensions)
        
        for i, node in enumerate(self.graph.nodes):
            self.graph.nodes[node]['position'] = position_matrix[i]
        
        return best_fitness
    
    def find_optimal_paths(self, source, target, n_paths=5):
        """Find multiple diverse optimal paths using swarm intelligence"""
        # Define a path fitness function
        def path_fitness(path_indices):
            # Convert continuous values to node indices
            n_nodes = len(self.graph.nodes)
            nodes = list(self.graph.nodes)
            
            # Extract path nodes
            path_length = min(len(path_indices) // 2, 20)  # Limit path length
            path = [source]
            
            for i in range(path_length):
                idx = int(path_indices[i] * n_nodes) % n_nodes
                node = nodes[idx]
                
                # Skip if node already in path or not connected
                if node in path or not self.graph.has_edge(path[-1], node):
                    continue
                
                path.append(node)
                
                # Stop if target reached
                if node == target:
                    break
            
            # If target not reached, penalize
            if path[-1] != target:
                path.append(target)
                return -1000  # Large penalty
            
            # Calculate path cost
            cost = 0
            for i in range(len(path) - 1):
                u, v = path[i], path[i+1]
                if self.graph.has_edge(u, v):
                    cost += self.graph.edges[u, v].get('weight', 1.0)
                else:
                    return -1000  # Invalid path
            
            # Calculate path diversity (for multiple paths)
            diversity_bonus = len(set(path))  # Favor paths with more unique nodes
            
            # Path fitness is negative cost plus diversity bonus
            return -cost + 0.1 * diversity_bonus
        
        # Setup a swarm optimizer for path finding
        path_optimizer = ChaosSwarmOptimizer(
            fitness_func=path_fitness,
            dimensions=40,  # Allow for paths up to 20 nodes
            swarm_size=self.swarm_size,
            bounds=[(0, 1) for _ in range(40)],
            use_chaos=self.use_chaos
        )
        
        # Run the optimizer
        best_position, _ = path_optimizer.optimize(iterations=50)
        
        # Extract the path from the best position
        n_nodes = len(self.graph.nodes)
        nodes = list(self.graph.nodes)
        
        path = [source]
        path_length = min(len(best_position) // 2, 20)
        
        for i in range(path_length):
            idx = int(best_position[i] * n_nodes) % n_nodes
            node = nodes[idx]
            
            if node not in path and self.graph.has_edge(path[-1], node):
                path.append(node)
                
                if node == target:
                    break
        
        if path[-1] != target:
            path.append(target)
        
        return path
    
    def optimize_network_flows(self, demand_pairs, iterations=50):
        """Optimize the entire network for multiple source-destination pairs"""
        # Define a fitness function for network flow optimization
        def network_flow_fitness(weight_factors):
            # Apply weight factors to the original edge weights
            modified_weights = {}
            for i, (u, v) in enumerate(self.graph.edges):
                idx = i % len(weight_factors)
                factor = 0.5 + weight_factors[idx]  # 0.5 to 1.5 range
                modified_weights[(u, v)] = self.edge_weights.get((u, v), 1.0) * factor
            
            # Calculate overall network performance
            total_cost = 0
            path_count = 0
            
            for source, target in demand_pairs:
                tryOllama integration and visualization
