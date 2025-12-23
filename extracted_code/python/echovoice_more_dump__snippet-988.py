"""Loads a HybridState from a file."""
with np.load(filename) as data:
    E = {k: v for k, v in zip(data['E_keys'], data['E_values'])}
    x = {k: v for k, v in zip(data['x_keys'], data['x_values'])}

print(f"\n[Memory Engine] System state loaded from {filename}")
return HybridState(E=E, x=x)

