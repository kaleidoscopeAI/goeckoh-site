def simulate_input():
    return input("Interact (type query): ").strip()

def ans_hash(agent_id):
    return hashlib.sha256(agent_id.encode()).hexdigest()[:8]  # Sim cryptographic ID

