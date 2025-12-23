"""Loads configuration from a JSON file."""
with open(config_path, 'r') as f:
    config_data = json.load(f)

for key, value in config_data.items():
    if hasattr(self, key):
        setattr(self, key, value)

