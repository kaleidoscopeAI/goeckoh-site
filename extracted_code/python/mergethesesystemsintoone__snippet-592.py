"""Generate amplitude pattern from input data."""
combined = str(sorted(data.items()))
hash_val = hash(combined)
rng = np.random.RandomState(hash_val)
return np.abs(rng.normal(0, 1, self.dimensions))

