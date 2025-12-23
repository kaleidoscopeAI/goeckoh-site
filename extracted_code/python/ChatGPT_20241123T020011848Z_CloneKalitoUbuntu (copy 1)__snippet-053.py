import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Node class to encapsulate behavior and state
class Node:
    def __init__(self, node_id, dna, position):
        self.node_id = node_id
        self.dna = dna
        self.position = position
        self.fitness = None
        self.logs = []

    def evaluate_fitness(self, target):
        self.fitness = -np.linalg.norm(self.dna - target)
        self.logs.append(f"Fitness evaluated: {self.fitness:.2f}")

    def mutate(self, mutation_rate):
        for i in range(len(self.dna)):
            if np.random.rand() < mutation_rate:
                self.dna[i] += np.random.normal(0, 0.1)
        self.dna = np.clip(self.dna, 0, 1)
        self.logs.append(f"Mutated DNA: {self.dna}")

    def replicate(self, mutation_rate, node_id_counter):
        child_dna = self.dna.copy()
        for i in range(len(child_dna)):
            if np.random.rand() < mutation_rate:
                child_dna[i] += np.random.normal(0, 0.1)
        child_position = self.position + np.random.uniform(-0.1, 0.1, size=2)
        child_position = np.clip(child_position, 0, 1)
        self.logs.append(f"Replicated to position: {child_position}")
        return Node(node_id_counter, dna=np.clip(child_dna, 0, 1), position=child_position)

# Parameters
num_generations = 50
mutation_rate = 0.2
initial_dna = np.random.rand(5)
nodes = [Node(node_id=1, dna=initial_dna, position=np.random.rand(2))]
node_id_counter = 2

# Dynamic target for fitness evaluation
dynamic_targets = [np.random.rand(5) + i * 0.1 for i in range(num_generations // 5)]

# Visualization setup
fig, axs = plt.subplots(2, 1, figsize=(10, 12))
scatter_ax = axs[0]
trend_ax = axs[1]

scatter_ax.set_xlim(0, 1)
scatter_ax.set_ylim(0, 1)
scatter_ax.set_title("Node Evolution and Adaptation")
scatter_ax.set_xlabel("X Position")
scatter_ax.set_ylabel("Y Position")
scatters = scatter_ax.scatter([], [], s=[], c=[], alpha=0.7)

trend_ax.set_title("Fitness and Replication Trends")
trend_ax.set_xlabel("Generation")
trend_ax.set_ylabel("Value")
fitness_line, = trend_ax.plot([], [], label="Average Fitness", color="blue")
replication_line, = trend_ax.plot([], [], label="Replication Count", color="green")
trend_ax.legend()

fitness_trends = []
replication_counts = []

def update(frame):
    global nodes, node_id_counter

    current_target = dynamic_targets[frame // (num_generations // len(dynamic_targets))]
    next_gen_nodes = []

    for node in nodes:
        node.evaluate_fitness(current_target)
        if np.random.rand() < 0.5:
            child_node = node.replicate(mutation_rate, node_id_counter)
            node_id_counter += 1
            next_gen_nodes.append(child_node)

    nodes = sorted([n for n in nodes if n.fitness is not None], key=lambda n: n.fitness, reverse=True)[:len(nodes) // 2]
    nodes.extend(next_gen_nodes)

    avg_fitness = np.mean([node.fitness for node in nodes if node.fitness is not None])
    fitness_trends.append(avg_fitness)
    replication_counts.append(len(next_gen_nodes))

    positions = np.array([node.position for node in nodes])
    if positions.size > 0:
        scatters.set_offsets(positions)
        sizes = [(1 + (node.fitness or 0)) * 100 for node in nodes]
        colors = [node.dna[:3] for node in nodes]
        scatters.set_sizes(sizes)
        scatters.set_facecolor(colors)

    fitness_line.set_data(range(len(fitness_trends)), fitness_trends)
    replication_line.set_data(range(len(replication_counts)), replication_counts)
    trend_ax.set_xlim(0, len(fitness_trends))
    trend_ax.set_ylim(0, max(max(fitness_trends, default=0), max(replication_counts, default=0)) + 1)

    return scatters, fitness_line, replication_line

def init():
    scatters.set_offsets([])
    fitness_line.set_data([], [])
    replication_line.set_data([], [])
    return scatters, fitness_line, replication_line

ani = FuncAnimation(fig, update, frames=num_generations, init_func=init, interval=500, repeat=False)

plt.tight_layout()
plt.show()

