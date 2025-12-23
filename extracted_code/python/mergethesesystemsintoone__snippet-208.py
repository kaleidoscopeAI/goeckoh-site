def __post_init__(self):
    self.harmonics = []
    self.interference_pattern = None
    self._update_interference()

def _update_interference(self):
    """Updates the interference pattern based on current state."""
    self.interference_pattern = np.outer(self.frequency, self.amplitude) * \
                                np.exp(1j * self.phase.reshape(-1, 1))

def interact(self, other: 'MemoryResonance') -> float:
    """Calculate interaction strength with another memory pattern."""
    interference = np.abs(np.sum(self.interference_pattern *
                                 np.conj(other.interference_pattern)))
    return float(interference)

def evolve(self, delta_time: float, field_state: np.ndarray, global_stats: Dict):
    """
    Evolves the memory pattern over time based on field interactions and global statistics.
    """
    # Calculate coupling strength with normalization
    total_field_energy = np.sum(np.abs(field_state)**2)
    field_coupling = np.sum(self.interference_pattern * np.conj(field_state)) / (total_field_energy + 1e-6)

    # Phase evolution with field influence
    phase_shift = np.angle(field_coupling) * delta_time
    self.phase += self.frequency * delta_time + phase_shift

    # Frequency modulation based on field resonance and success
    resonance_strength = np.abs(field_coupling)
    
    # Access pattern matching success from global statistics
    success_rate = global_stats.get("pattern_matching_success", {}).get(self.pattern_id, 0.5)  # Default to 0.5 if not found

    # Modify frequency shift based on success rate
    freq_shift = resonance_strength * np.sin(self.phase) * 0.05 * (success_rate - 0.5) # Now influenced by success_rate
    self.frequency += freq_shift

    # Amplitude modulation based on resonance strength and validation from other resonances
    amplitude_modulation = (1.0 + np.tanh(resonance_strength - 0.5) * delta_time)
    
    # Consider validation from other resonances (simplified example)
    validation_factor = 1.0
    for other_id, other_resonance in global_stats.get("resonances", {}).items():
        if other_id != self.pattern_id:
            validation_factor += self.interact(other_resonance) * 0.1  # Example: small influence from other resonances

    self.amplitude *= amplitude_modulation * validation_factor
    
    # Harmonic influence and generation
    self._update_harmonics(resonance_strength, global_stats)

    # Update interference pattern
    self._update_interference()

    # Adapt thresholds based on global statistics
    self._adapt_thresholds(global_stats)

def _update_harmonics(self, resonance_strength: float, global_stats: Dict):
  """Updates harmonics based on resonance strength and interactions with other patterns."""

  # Influence from other harmonics
  for other_id, other_resonance in global_stats.get("resonances", {}).items():
      if other_id != self.pattern_id:
          for harmonic in other_resonance.harmonics:
              if self._is_harmonic_related(self.frequency, harmonic):
                  # Increase strength of related harmonics
                  if harmonic not in self.harmonics:
                      self.harmonics.append(harmonic)
                  else:
                    self.harmonics[self.harmonics.index(harmonic)] = harmonic

  # Generate new harmonics
  if len(self.harmonics) < 5 and np.random.random() < resonance_strength * 0.2:
      new_harmonic = np.mean(self.frequency) * (1.5 + np.random.random())
      if new_harmonic not in self.harmonics:
          self.harmonics.append(new_harmonic)

  # Remove harmonics that are not related to any other pattern
  self.harmonics = [h for h in self.harmonics if any(self._is_harmonic_related(h, other_resonance.frequency)
                                                     for other_id, other_resonance in global_stats.get("resonances", {}).items()
                                                     if other_id != self.pattern_id)]

def _is_harmonic_related(self, freq1, freq2) -> bool:
  """Checks if two frequencies are harmonically related."""
  ratio = max(freq1, freq2) / min(freq1, freq2)
  return abs(round(ratio) - ratio) < 0.1  # Tolerance for harmonic relationship

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

    # Apply bounds with a learning rate
    self.frequency = np.clip(self.frequency, self.frequency_bounds[0], self.frequency_bounds[1])
    self.amplitude = np.clip(self.amplitude, self.amplitude_bounds[0], self.amplitude_bounds[1])
