"""
Represents the genetic code of a node, influencing its behavior and capabilities.
"""
learning_rate: float = 0.1
mutation_rate: float = 0.01
energy_efficiency: float = 1.0
memory_capacity: int = 100
initial_energy: float = 50.0
replication_threshold: float = 0.8
min_memory_for_replication: int = 50
energy_consumption_rate: float = 0.1

def mutate(self):
    """
    Mutates the genetic code, introducing variations in traits.
    """
    new_code = GeneticCode()
    for trait in vars(self):
        if trait == "mutation_rate":
            continue  # Mutation rate itself can't be mutated
        original_value = getattr(self, trait)
        if isinstance(original_value, float):
            mutation = original_value * random.uniform(-self.mutation_rate, self.mutation_rate)
            new_value = max(0.01, original_value + mutation)  # Prevent traits from being too low
            setattr(new_code, trait, new_value)
        elif isinstance(original_value, int):
            mutation = int(original_value * random.uniform(-self.mutation_rate, self.mutation_rate))
            new_value = max(1, original_value + mutation)  # Prevent traits from being too low
            setattr(new_code, trait, new_value)

    return new_code

def combine(self, other_dna):
    """
    Combines traits from two GeneticCode instances to create a new one.

    Args:
        other_dna (GeneticCode): Another GeneticCode instance to combine traits with.

    Returns:
        GeneticCode: A new GeneticCode instance with combined traits.
    """
    new_code = GeneticCode()
    for trait in vars(self):
        if trait == "mutation_rate":
            # Mutation rate itself is not combined, it remains as is
            setattr(new_code, trait, self.mutation_rate)
        else:
            # Randomly choose between traits of the two parent DNAs
            if random.random() < 0.5:
                setattr(new_code, trait, getattr(self, trait))
            else:
                setattr(new_code, trait, getattr(other_dna, trait))

    return new_code



