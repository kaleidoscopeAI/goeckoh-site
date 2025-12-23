traits: List[float]
def mutate(self) -> 'NodeDNA':
    # Evolution logic: gentle drift in traits
    new_traits = [t + random.uniform(-0.05, 0.05) for t in self.traits]
    return NodeDNA(traits=new_traits)

