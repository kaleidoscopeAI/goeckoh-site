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

