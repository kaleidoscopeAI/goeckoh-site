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
