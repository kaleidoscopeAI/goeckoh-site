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

