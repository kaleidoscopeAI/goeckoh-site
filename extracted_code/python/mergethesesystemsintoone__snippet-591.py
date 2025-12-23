"""
Store information as a resonance pattern in the field.
Returns the pattern_id of the stored resonance.
"""
# Generate unique frequency components from data
frequency = self._hash_to_frequency(str(data))
amplitude = self._generate_amplitude(data)
phase = self._generate_phase(data)

pattern_id = str(uuid.uuid4())
resonance = MemoryResonance(
    pattern_id=pattern_id,
    frequency=frequency,
    amplitude=amplitude,
    phase=phase
)

self.resonances[pattern_id] = resonance
self._update_field_state()
return pattern_id

