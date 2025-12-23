def enforce_first_person(text: str) -> str:
    if not text: return ""
    t = text.strip().strip('"').strip("'")
    patterns = [
        (r"\byou are\b", "I am"), (r"\byou're\b", "I'm"), (r"\byou were\b", "I was"),
        (r"\byou'll\b", "I'll"), (r"\byou've\b", "I've"), (r"\byour\b", "my"),
        (r"\byours\b", "mine"), (r"\byourself\b", "myself"), (r"\byou\b", "I"),
    ]
    for pattern, repl in patterns:
        t = re.sub(pattern, repl, t, flags=re.IGNORECASE)
    return t

def hash_embedding(text: str, dim: int) -> np.ndarray:
    vec = np.zeros(dim, dtype=np.float32)
    if not text: return vec
    for tok in text.lower().split():
        h = hashlib.sha256(tok.encode("utf-8")).digest()
        idx = struct.unpack("Q", h[:8])[0] % dim
        sign = 1.0 if (struct.unpack("I", h[8:12])[0] % 2 == 0) else -1.0
        vec[idx] += sign
    norm = np.linalg.norm(vec) + 1e-8
    return vec / norm

