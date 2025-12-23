"""Reads the universal master key from the key file."""
try:
    with open(path, 'r') as f:
        # Read the key and strip any whitespace/newlines
        return f.read().strip()
except FileNotFoundError:
    print(f"FATAL ERROR: The Genesis Key file ('{path}') was not found.")
    print("Cannot issue any master commands without the key.")
    exit(1)

