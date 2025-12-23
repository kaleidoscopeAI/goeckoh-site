    # phase evoluition from signal and field
    phase_shift = np.angle(field_coupling) * delta_time # update phase by interaction with field over given timeframe
    self.phase += self.frequency * delta_time + phase_shift

      # Frequency shift related to signal strengh and frequency in direction or phase/interference vector
    resonance_strength = np.abs (field_coupling) # strenght based on field enery overlap
    success_rate = global_stats.get ("pattern_matching_success", {}).get(self.pattern_id, 0.5)
    freq_shift = resonance_strength * np.sin (self.phase) * 0.05 * (success_rate - 0.5) # Modified by pattern success (0-1 scale, shifting more if lower or higher)
    self.frequency += freq_shift

      # Ampltiude of vector based on reinforcement from existing pattern matching and validation score from interaction with field (feedback effect for patterns)
    amplitude_modulation = (1.0 + np.tanh(resonance_strength - 0.5) * delta_time) # smooth step in reinforcement. if pattern is matched better
      
      # Strength of the patter gets influenced by the interaction between other memories as well based on the existing state.
    validation_factor = 1.0  # Placehodler in original concept. 
    for other_id, other_resonance in global_stats.get ("resonances",{}).items():
        if other_id != self.pattern_id:
           validation_factor += self.interact (other_resonance) * 0.1  # Example, using dot for inner correlation product of field states to determine coupling

      # Use Validation feedback score or a generic "time to decay"" factor for time aware modulation

    self.amplitude *= amplitude_modulation * validation_factor
  
   # Harmonic interaction using harmonic coupling function

    self._update_harmonics (resonance_strength, global_stats) # Use other resonance patterns to reinforce own patterns using harmonic relationships


    # Dynamic adaptation based on field state

    self._adapt_thresholds (global_stats) #Adapt thresoholds to match with overall states.

      #Update interference patterns for visualization
    self._update_interference()

def _update_harmonics (self, resonance_strength: float, global_stats: Dict):
     """ Update frequency harmonics based on relationship or by simple generation methods.

      - uses resonance stenght, and interacts and amplifies based on similar nodes and removes if it does not relate
        # parameters - field_stat and data.
        # return nothing is handled inplace on the resonnace obj.
     """
      # check the exisiting resonances of all current resonances nodes influence based on similar frequencies with harmonic functions of sine waves to enforce.

      #Harmonic reinforcement based on matching harmonics
     for other_id, other_resonance in global_stats.get("resonances", {}).items():
        if other_id != self.pattern_id:
              for harmonic in other_resonance.harmonics:
                 if self._is_harmonic_related(self.frequency, harmonic):
                   # increase strength if other harmonic exists.
                   if harmonic not in self.harmonics:
                       self.harmonics.append(harmonic)
                   else:
                     self.harmonics[self.harmonics.index(harmonic)] = harmonic  # Replace value if it exists

        # Genrate new harmonic with weighted average if strenght increases
     if len (self.harmonics) < 5 and random.random () < resonance_strength * self.harmonic_influence_factor: # Increase the randomness threshold by amplitude

          new_harmonic = np.mean(self.frequency) * (1.5 + random.random()) # Calculate average freqs of self
          if new_harmonic not in self.harmonics:
             self.harmonics.append(new_harmonic) # Appends if unique

     # Clean harmonics with very low frequency matching of 1 or 0. only harmonics that connect get updated. If a hormic doesn't relate over time then remove it
     self.harmonics = [h for h in self.harmonics if any (self._is_harmonic_related (h, other_resonance.frequency)
                      for other_id, other_resonance in global_stats.get ("resonances", {}).items()
                     if other_id != self.pattern_id) ]
def _is_harmonic_related(self, freq1, freq2) -> bool:
  """ Check if 2 freqs are in phase/harmonic (for simplicity using  near ratios or factors of each other"""
  ratio = max(freq1, freq2) / (min (freq1, freq2) + le-6) # calculate a ration of high/ low of frequencie (add a small scaler for stability)
  return abs (round(ratio) ratio) < 0.1  # simple tolerance for harmonic

def _adapt_thresholds(self, global_stats: Dict):
  """
     Adapts thresholds using overall field to update, frequency and amplitue range limits in current memory.
    """
  all_frequencies = [r.frequency for r in global_stats.get ("resonances", {}).values ()]
  all_amplitudes = [r.amplitude for r in global_stats.get ("resonances", {}).values ()]

    
  if all_frequencies:
    avg_freq = np.mean (all_frequencies)
    std_freq = np.std (all_frequencies)
    self.frequency_bounds = (
            max (-10, avg_freq 3 * std_freq), # dynamicly adpat the frequncy by some dynamic percentage
            min (10, avg_freq + 3 * std_freq)
        )
        
  if all_amplitudes:
    avg_amp = np.mean (all_amplitudes)
    std_amp = np.std (all_amplitudes)
    self.amplitude_bounds = (
         max (0.01, avg_amp 3 * std_amp),
         min (20, avg_amp + 3 * std_amp) # Scale and bound for max and minimum based on environment.
    )

  # Apply new bounds, so they stay within our boundaries
  self.frequency = np.clip (self.frequency, self.frequency_bounds [0], self.frequency_bounds [1])
  self.amplitude = np.clip(self.amplitude, self.amplitude_bounds[0], self.amplitude_bounds [1])
