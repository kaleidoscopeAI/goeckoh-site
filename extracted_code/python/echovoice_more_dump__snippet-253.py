import numpy as np

def sensor_grad(kappa, adc_raw):
    return np.gradient(adc_raw) * kappa  # Real ∂I/∂t = ∇ · (κ ADC)

