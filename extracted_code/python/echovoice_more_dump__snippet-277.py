def save_state(state: HybridState, filename="cognitive_state.npz"):
    """Saves the HybridState to a compressed .npz file."""
    # We can't save the raw dictionaries, so we convert them to savable arrays
    E_keys = np.array(list(state.E.keys()))
    E_values = np.array([state.E[k] for k in E_keys])
    
    x_keys = np.array(list(state.x.keys()))
    x_values = np.array([state.x[k] for k in x_keys])
    
    np.savez(filename, E_keys=E_keys, E_values=E_values, x_keys=x_keys, x_values=x_values)
    print(f"\n[Memory Engine] System state saved to {filename}")

def load_state(filename="cognitive_state.npz") -> HybridState:
    """Loads a HybridState from a file."""
    with np.load(filename) as data:
        E = {k: v for k, v in zip(data['E_keys'], data['E_values'])}
        x = {k: v for k, v in zip(data['x_keys'], data['x_values'])}
    
    print(f"\n[Memory Engine] System state loaded from {filename}")
    return HybridState(E=E, x=x)

