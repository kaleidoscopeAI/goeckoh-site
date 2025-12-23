# Update function for animation
def update(frame):
    global nodes, node_id_counter

    # Determine current target
    current_target = dynamic_targets[frame // (num_generations // len(dynamic_targets))]
    next_gen_nodes = []

    # Evaluate fitness and replicate nodes
    for node in nodes:
        node.evaluate_fitness(current_target)
        if np.random.rand() < 0.5:  # 50% chance to replicate
            child_node = node.replicate(mutation_rate, node_id_counter)
            node_id_counter += 1
            next_gen_nodes.append(child_node)

    # Competitive selection: Keep top 50% nodes based on fitness
    nodes = sorted([n for n in nodes if n.fitness is not None], key=lambda n: n.fitness, reverse=True)[:len(nodes) // 2]
    nodes.extend(next_gen_nodes)

    # Update trends
    avg_fitness = np.mean([node.fitness for node in nodes if node.fitness is not None])
    fitness_trends.append(avg_fitness)
    replication_counts.append(len(next_gen_nodes))

    # Update scatter plot
    positions = np.array([node.position for node in nodes])
    sizes = [(1 + (node.fitness or 0)) * 100 for node in nodes]
    colors = [node.dna[:3] for node in nodes]  # Use first 3 DNA values for RGB color
    scatters.set_offsets(positions)
    scatters.set_sizes(sizes)
    scatters.set_color(colors)

    # Update trend lines
    fitness_line.set_data(range(len(fitness_trends)), fitness_trends)
    replication_line.set_data(range(len(replication_counts)), replication_counts)
    trend_ax.set_xlim(0, len(fitness_trends))
    trend_ax.set_ylim(0, max(max(fitness_trends), max(replication_counts)) + 1)

    return scatters, fitness_line, replication_line

# Initialize function for animation
def init():
    scatters.set_offsets([])
    fitness_line.set_data([], [])
    replication_line.set_data([], [])
    return scatters, fitness_line, replication_line

