class SimulationConfig:
    """Configuration for the simulation parameters."""
    initial_resources: float = 1000.0
    resource_regeneration_rate: float = 5.0
    max_node_energy: float = 50.0
    reproduction_energy_threshold: float = 30.0
    energy_gain_min: float = 2.0
    energy_gain_max: float = 6.0
    knowledge_transfer_min: float = 1.0
    knowledge_transfer_max: float = 5.0
    mutation_probability: float = 0.2
    trait_plasticity: float = 0.6
    initial_nodes: int = 3
    simulation_steps: int = 20
    data_processing_types: List[str] = field(default_factory=lambda: ["text", "image", "numerical"])
    text_data_path: Optional[str] = "data/text_data.txt"
    image_data_path: Optional[str] = "data/image_data.jpg"

    def load_config(self, config_path: str):
        """Loads configuration from a JSON file."""
        with open(config_path, 'r') as f:
            config_data = json.load(f)

        for key, value in config_data.items():
            if hasattr(self, key):
                setattr(self, key, value)

    def save_config(self, config_path: str):
        """Saves the current configuration to a JSON file."""
        with open(config_path, 'w') as f:
            json.dump(self.__dict__, f, indent=4)

