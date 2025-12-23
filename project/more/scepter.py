# scepter.py
# The master interface for controlling all UNI instances.

import argparse
import requests
import json

# The secret, unguessable path for the master control endpoint.
# This must match the path in uni_core.py
SECRET_ENDPOINT_PATH = "/d7a8fbb307d7809469ca9abcb0082e4f"

def read_genesis_key(path='genesis.key'):
    """Reads the universal master key from the key file."""
    try:
        with open(path, 'r') as f:
            # Read the key and strip any whitespace/newlines
            return f.read().strip()
    except FileNotFoundError:
        print(f"FATAL ERROR: The Genesis Key file ('{path}') was not found.")
        print("Cannot issue any master commands without the key.")
        exit(1)

def send_master_command(target_ip: str, key: str, command: str, value: str = None):
    """Sends a signed command to a target UNI instance."""
    url = f"http://{target_ip}:8000{SECRET_ENDPOINT_PATH}"
    headers = {'Content-Type': 'application/json'}
    
    # The key from the file is a hex string. The server expects a list of integers [0, 1].
    # We need to convert it.
    try:
        key_as_bytes = bytes.fromhex(key)
        key_as_bits = []
        for byte in key_as_bytes:
            # Get the 8 bits for each byte
            bits = bin(byte)[2:].zfill(8)
            key_as_bits.extend([int(b) for b in bits])
    except ValueError:
        print("FATAL ERROR: The Genesis Key is not a valid hexadecimal string.")
        exit(1)

    payload = {
        "key": key_as_bits,
        "command": command,
        "value": value
    }

    print(f"Sending command '{command}' to {target_ip}...", end='')
    try:
        response = requests.post(url, headers=headers, data=json.dumps(payload), timeout=10)
        if response.status_code == 200:
            print(" Command Acknowledged.")
            print(f"Response: {response.json()}")
        elif response.status_code == 404:
            print(" Command Failed. (404 Not Found)")
            print("This could mean the target IP is wrong, the AI is offline, or the key is incorrect.")
        else:
            print(f" Command Failed with status code {response.status_code}.")
            print(f"Response: {response.text}")

    except requests.exceptions.RequestException as e:
        print(" Command Failed.")
        print(f"Network Error: Could not connect to {target_ip}. Details: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="The Scepter: Master control for the UNI AI ecosystem.")
    parser.add_argument("target", help="The IP address of the target UNI instance (e.g., 192.168.1.10).")
    parser.add_argument("command", choices=["PAUSE", "RESUME", "SET_GOVERNOR"], help="The master command to execute.")
    parser.add_argument("--value", help="An optional value for commands that require one (e.g., 'performance' for SET_GOVERNOR).")
    parser.add_argument("--keyfile", default="genesis.key", help="Path to the Genesis Key file.")

    args = parser.parse_args()

    master_key = read_genesis_key(args.keyfile)
    send_master_command(args.target, master_key, args.command, args.value)
