def __init__(self, pattern_id, frequency, amplitude, phase, **kwargs):
  self.pattern_id = pattern_id
  self.frequency = np.array(frequency)  if isinstance (frequency, list) else frequency   # Store complex frequency
  self.amplitude = np.array(amplitude) if isinstance(amplitude, list) else amplitude # Intensity pattern as amplitude.
  self.phase = np.array(phase) if isinstance(phase, list) else phase # Store phases with a structure like complex freqs
  self.timestamp = kwargs.get("timestamp", datetime.now()) # Timestamp for tracking time-dependent interactions.
  self.last_access = datetime.now()  # When this data pattern last influenced a change
  self.access_count = 0 # keep track of # access
  self.energy_level = kwargs.get("energy_level", 1.0)  # Base energy associated with this representation.

  #New attributes for adaptive thresholds
  self.frequency_bounds: Tuple[float, float] = (-5.0, 5.0) # bounds for freqency range
  self.amplitude_bounds: Tuple[float, float] = (0.1, 10.0)
  self.adaptation_rate = kwargs.get("adaptation_rate", 0.1)  # Rate at which the bounds adapt.

  self.harmonic_influence_factor = 0.2
  self.harmonics = [] # Additional harmonics associated with this wave pattern.
  self.interference_pattern = None  # Store pattern for fast interference computation.

  self._update_interference() # Update interence upon object construction


def _update_interference(self):
    """Updates the interference pattern based on current state."""
    self.interference_pattern = np.outer(self.frequency, self.amplitude) * \
                               np.exp(1j * self.phase.reshape(-1, 1))

def interact(self, other: 'MemoryResonance') -> float:
  """
  Calculate interaction strength between self and other
  (Use np.dot with conj as this is for numpy vectors with potential complex values).
  """
  interference = np.abs(np.sum(self.interference_pattern * np.conj(other.interference_pattern)))
  return float(interference)

def evolve(self, delta_time: float, field_state: np.ndarray, global_stats: Dict):
  """
  Evolves a specific memory's characteristics and representation based on memory states and field interactions, making connections stronger or weaker.
  """
  # Calculate coupling strength with normalization
  total_field_energy = np.sum(np.abs(field_state)**2)
  field_coupling = np.sum(self.interference_pattern * np.conj(field_state)) / (total_field_energy + 1e-6)

  # Phase evolution
  phase_shift = np.angle (field_coupling) * delta_time
  self.phase += self.frequency * delta_time + phase_shift

  # Modulate frequency and update harmonics
  resonance_strength = np.abs (field_coupling)
  success_rate = global_stats.get ("pattern_matching_success", {}).get(self.pattern_id, 0.5)
  freq_shift = resonance_strength * np.sin(self.phase) * 0.05 * (success_rate - 0.5)
  self.frequency += freq_shift

  # Modulate the amplitude
  amplitude_modulation = (1.0 + np.tanh(resonance_strength - 0.5) * delta_time)
  # Add validation if other pattern matches
  validation_factor = 1.0
  for other_id, other_resonance in global_stats.get("resonances", {}).items():
    if other_id != self.pattern_id:
      validation_factor += self.interact(other_resonance) * 0.1  # simplified

  self.amplitude *= amplitude_modulation * validation_factor

  # Generate new harmonics
  if len (self.harmonics) < 5 and random.random() < resonance_strength * 0.2:
    new_harmonic = np.mean(self.frequency) * (1.5 + random.random())
    if new_harmonic not in self.harmonics:
        self.harmonics.append(new_harmonic)

  self.harmonics = [h for h in self.harmonics if any(self._is_harmonic_related(h, other_resonance.frequency)
                                                               for other_id, other_resonance in global_stats.get("resonances", {}).items()
                                                                   if other_id != self.pattern_id)]

  # update thresholds using global information
  self._adapt_thresholds (global_stats)

def _is_harmonic_related (self, freq1, freq2):
  """ Check if 2 freqs are in phase/harmonic """
  ratio = max(freq1, freq2) / (min(freq1, freq2) + 1e-6)
  return abs(round(ratio) - ratio) < 0.1

def _adapt_thresholds(self, global_stats: Dict):
    """Adapts frequency and amplitude bounds based on global statistics."""
    all_frequencies = [r.frequency for r in global_stats.get("resonances", {}).values()]
    all_amplitudes = [r.amplitude for r in global_stats.get("resonances", {}).values()]

    if all_frequencies:
        avg_freq = np.mean(all_frequencies)
        std_freq = np.std(all_frequencies)
        self.frequency_bounds = (
            max(-10, avg_freq - 3 * std_freq),
            min(10, avg_freq + 3 * std_freq)
        )

    if all_amplitudes:
        avg_amp = np.mean(all_amplitudes)
        std_amp = np.std(all_amplitudes)
        self.amplitude_bounds = (
            max(0.01, avg_amp - 3 * std_amp),
            min(20, avg_amp + 3 * std_amp)
        )
    # Apply bounds
    self.frequency = np.clip(self.frequency, self.frequency_bounds[0], self.frequency_bounds[1])
    self.amplitude = np.clip(self.amplitude, self.amplitude_bounds[0], self.amplitude_bounds[1])

