class BitRegister:
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

class SimulatedHardware:
    def __init__(self, adc_channels: int = 16):
        self.cpu_register = BitRegister("CPU_REG", size=256)
        self.gpio = BitRegister("GPIO", size=64)
        self.adc = np.zeros(adc_channels, dtype=float)
        self.temp_C = 35.0
        self.freq_GHz = 1.2

    def poll_sensors(self):
        # realistic sensor noise and drift
        self.adc += np.random.randn(len(self.adc)) * 0.005
        self.adc = np.clip(self.adc, -5.0, 5.0)
        self.temp_C += (self.freq_GHz - 1.0) * 0.02 + np.random.randn() * 0.01

    def set_frequency(self, ghz: float):
        self.freq_GHz = float(max(0.2, min(ghz, 5.0)))

    def as_status(self):
        return {
            "freq_GHz": round(self.freq_GHz, 3),
            "temp_C": round(self.temp_C, 3),
            "cpu_reg": self.cpu_register.as_bitstring(),
            "gpio": self.gpio.as_bitstring(),
            "adc": [round(float(x), 4) for x in self.adc.tolist()],
        }

