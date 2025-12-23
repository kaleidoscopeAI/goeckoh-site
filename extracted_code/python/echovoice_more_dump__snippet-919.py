class QuantumBit:
    def __init__(self, p_real=0.5, p_imag=0.5):
        self.real = p_real
        self.imag = p_imag

    def measure(self):
        prob = self.real ** 2 + self.imag ** 2
        return 1 if random.random() < prob else 0

    def entangle(self, other):
        self.real, other.real = (self.real + other.real) / 2, (self.real + other.real) / 2
        self.imag, other.imag = (self.imag + other.imag) / 2, (self.imag + other.imag) / 2

