class MemoryResonance:
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

class MemoryField:
  """
  A class to simulate a dynamic memory space using resonance patterns and tensor fields.
  The field itself acts like a collective short and long term memory in conjuction
  """

  def __init__(self, dimensions: int = 64, decay_rate: float = 0.1):
    self.dimensions = dimensions
    self.decay_rate = decay_rate
    self.resonances: Dict[str, MemoryResonance] = {} # Dictionary to track pattern specific state information keyed by id of resonance (not the text)
    self.field_state = np.zeros((dimensions, dimensions), dtype=np.complex128) # Overall energy or knowledge state (matrix)
    self.interaction_history: List[Tuple[str, str, float]] = []
    self.energy_threshold = 0.01
  
  def store (self, data: Dict, position: Optional [np.ndarray] = None) -> str:
      """
        Stores and transforms incoming data into the system
        : param  data : Dictionary with key properties describing the pattern, may include metadata such as category etc.
        : return id of stored resonance if valid

        Implements pattern extraction to identify features to capture in resonant representations in 2d numpy fields
      """
      frequency = self._hash_to_frequency (str(data)) # Transforms Data (numerical vector) into an indexable item in Memory (vector frequency)
      amplitude = self._generate_amplitude (data) # Returns vector based on data input
      phase = self._generate_phase(data) # Maps patterns to time dimension. (vector complex space for frequency)
      pattern_id = str(uuid.uuid4())
      resonance = MemoryResonance(
      pattern_id=pattern_id, # unique id for each generated memory/pattern
      frequency=frequency, # unique frequency profile to represent patten characteristics/meaning
      amplitude=amplitude,
      phase=phase)
      self.resonances[pattern_id] = resonance # store it
      self._update_field_state()
      return pattern_id

  def retrieve (self, pattern: Dict) -> List [Tuple[str, float, Dict]]:
        """Retrieve patterns from resonant matching

         Returns a list of tuple where [id: float resonance: Dictionary representation of pattern)

        The data is transformed into resonance parameters then matched against existing nodes by similarity strength between frequency patterns . returns data that passes an energy treshold.
      """

        query_resonance = MemoryResonance(
          pattern_id="query", # This allows us to easily debug when looking for certain memories using logs etc..
          frequency=self._hash_to_frequency(str(pattern)), # Transforms query parameters into memory accessible frequencies
          amplitude=self._generate_amplitude(pattern), # returns matrix from given frequency components.
          phase = self._generate_phase(pattern),
          )

        results = []
        for pattern_id, resonance in self.resonances.items():
            interaction_strength = resonance.interact (query_resonance) # returns float between 0 - 1 indicating strenghth of relationship with data.
            if interaction_strength > self.energy_threshold:
              resonance.last_access = datetime.now()
              resonance.access_count += 1
              results.append((pattern_id, interaction_strength, self._reconstruct_data (resonance)))

        return sorted(results, key=lambda x: x[1], reverse=True)

  def _hash_to_frequency (self, data_str: str) -> np.ndarray:
        """Transforms input into complex frequency based on hashing """
        hash_value = int (hashlib.sha1 (data_str.encode()).hexdigest(), 16) # hash input data
        rng = np.random.RandomState (hash_value) # initilized a pseudo-random vector for this specific piece of data
        return rng.normal(0,1,self.dimensions)

  def _generate_amplitude (self, data: Dict) -> np.ndarray:
      """Transforms data into a amplitude using a simlar hashing strategy as in hash to freqs"""
      combined = str (sorted(data.items()))
      hash_value = int (hashlib.md5 (combined.encode()).hexdigest(), 16)
      rng = np.random.RandomState(hash_value)
      return np.abs(rng.normal(0, 1, self.dimensions))
    
  def _generate_phase (self, data: Dict) -> np.ndarray:
        """Transform data into a matrix of phases based on a simplified hashing method"""
        combined = str (sorted(data.items()))
        hash_val = int (hashlib.shake_256 (combined.encode()).hexdigest (256), 16)
        rng = np.random.RandomState (hash_val)
        return rng.uniform(0, 2 * np.pi, self.dimensions) # creates unique 2d vectors

  def _update_field_state(self):
    """Updates the field based on present resonances"""
    self.field_state.fill(0) # Reset field before every update
    
    for resonance in self.resonances.values():
      resonance._update_interference()
      self.field_state += resonance.interference_pattern  # Add each memory interference pattern into field state

        # Apply energy decay to resonances
    for resonance in self.resonances.values():
        time_delta = (datetime.now() - resonance.timestamp).total_seconds()
        resonance.energy_level *=
