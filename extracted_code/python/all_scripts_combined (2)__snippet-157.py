def __init__(self, name: str, size: int = 128):
    self.name = name
    self.size = size
    self.bits = np.zeros(size, dtype=np.uint8)
    self.noise_rate = 2e-6
    self.lock = Lock()

def write_int(self, value: int):
    with self.lock:
        for i in range(self.size):
            self.bits[i] = (value >> i) & 1

def read_int(self) -> int:
    with self.lock:
        if random.random() < self.noise_rate:
            idx = random.randrange(self.size)
            self.bits[idx] ^= 1
        out = 0
        for i in range(self.size):
            out |= int(self.bits[i]) << i
        return out

def set_bit(self, idx: int, val: int):
    with self.lock:
        self.bits[idx % self.size] = 1 if val else 0

def get_bit(self, idx: int) -> int:
    with self.lock:
        return int(self.bits[idx % self.size])

def as_bitstring(self) -> str:
    with self.lock:
        return ''.join(str(int(b)) for b in self.bits[::-1])

