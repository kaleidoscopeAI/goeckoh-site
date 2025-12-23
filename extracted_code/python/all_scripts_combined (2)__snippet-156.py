# deterministic hashing into bytes array using SHAKE-like approach without external libs
import hashlib
h = hashlib.blake2b(digest_size=32)
h.update(s.encode("utf8"))
base = h.digest()
out = bytearray()
i = 0
while len(out) < length:
    h2 = hashlib.blake2b(digest_size=32)
    h2.update(base)
    h2.update(bytes([i]))
    out.extend(h2.digest())
    i += 1
return bytes(out[:length])

