# Remove any unsupported hash types from the mapping. If this leaves no
# supported hashes, return None
if hashes is None:
    return None
hashes = {n: v for n, v in hashes.items() if n in _SUPPORTED_HASHES}
if not hashes:
    return None
return hashes


