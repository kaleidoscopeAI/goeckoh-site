import hashlib
h = hashlib.sha256((salt + s).encode("utf-8", "ignore")).digest()
return int.from_bytes(h[:8], "little")

