# Core Node and DNA Classes

class OrganicCore:
    def __init__(self, node_id, dna):
        self.node_id = node_id
        self.dna = dna
        self.memory = {}  # Memory to store experiences with weights

    def learn(self, experience, impact=1):
        # Add experience to memory with a weight (impact level)
        if experience in self.memory:
            self.memory[experience] += impact
        else:
            self.memory[experience] = impact
        self.adapt()

    def adapt(self):
        # Adjust DNA based on weighted experiences in memory
        for experience, weight in self.memory.items():
            self.dna.evolve_from_experience(experience, weight)

    def replicate(self):
        # Replicate with slight DNA adaptation
        new_dna = self.dna.mutate()
        return OrganicCore(node_id=self.node_id + "_child", dna=new_dna)


class NodeDNA:
    def __init__(self, traits):
        self.traits = traits

    def mutate(self):
        # Generate a variant of DNA for child nodes
        return NodeDNA(traits=[trait + self.variation() for trait in self.traits])

    def evolve_from_experience(self, experience, weight):
        # Adjust DNA traits based on experiences and weights
        self.traits = [trait + (experience * 0.1 * weight) for trait in self.traits]

    def variation(self):
        # Simple placeholder for DNA variation during mutation
        return 0.1

# Environment Class

class Environment:
    def __init__(self):
        self.resources = 100  # Example resource limit

    def provide_resources(self, node):
        # Simulate providing resources to a node
        if self.resources > 0:
            self.resources -= 10
            return 10  # Amount of resources provided
        return 0

    def adjust_environment(self, feedback):
        # Adjusts resources based on node success or resource levels
        if feedback == "success":
            self.resources += 5  # Positive feedback replenishes environment resources

# Simulation

# Create environment and initial root node
environment = Environment()
root_node = OrganicCore(node_id="root", dna=NodeDNA(traits=[1, 2, 3]))

# Simple simulation loop for 5 cycles
for cycle in range(5):
    # Node learns from experience (simulate experience value with cycle number)
    root_node.learn(experience=cycle, impact=cycle % 2 + 1)

    # Node replicates to create a child node
    child_node = root_node.replicate()

    # Environment interaction
    resources_received = environment.provide_resources(root_node)
    environment.adjust_environment("success" if resources_received > 0 else "low")

    # Output state for observation
    print(f"Cycle {cycle + 1}: Node {root_node.node_id} traits: {root_node.dna.traits}")
    print(f"Resources received: {resources_received}, Environment resources remaining: {environment.resources}")

