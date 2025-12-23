def sensor_grad(kappa, adc_raw):
    """Simulates the sensor gradient logic (∂I/∂t = ∇ · (κ ADC))"""
    if not adc_raw:
        return [0.0]
    # Simple discrete gradient calculation
    return np.gradient(np.array(adc_raw)) * kappa

