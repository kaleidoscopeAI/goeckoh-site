class MemoryResonance:
    def __init__(self, pattern_id, frequency, amplitude, phase, **kwargs):
      self.pattern_id = pattern_id
      self.frequency = np.array(frequency)  if isinstance (frequency, list) else frequency  # Store complex frequency
      self.amplitude = np.array(amplitude) if isinstance(amplitude, list) else amplitude  # Intensity pattern as amplitude.

