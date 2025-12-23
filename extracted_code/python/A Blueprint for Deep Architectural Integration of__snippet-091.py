/// Update curiosity bit using performance feedback and external drive,
/// implemented as a boolean threshold function.
pub fn update_curiosity_bit(
    &mut self,
    performance_feedback: f32,
    external_drive: f32,
) {
    let input_val = self.decay_rho * (if self.curiosity_bit { self.lambda } else { 0.0})
        - self.sigma * performance_feedback
        + self.beta * external_drive;

    self.curiosity_bit = input_val > self.threshold_theta;
}

/// Bitwise routing function: given input bitvectors from connected engines,
/// compute output bits for this engine via bitwise logical operations.
pub fn route_and_integrate(
    &self,
    engine_inputs: &[Vec<u64>], // input bit vectors from engines
    engine_index: usize,
) -> Vec<u64> {
    // Initialize output vector with zeros (same length as input vectors)
    let mut output_bits = vec![0u64; engine_inputs[^55_0].len()];

    for (src_index, input_bits) in engine_inputs.iter().enumerate() {
        if self.routing_matrix[engine_index][src_index] {
            // Example: combine inputs with OR; sophisticated logic can replace this
            for (i, &bits) in input_bits.iter().enumerate() {
                output_bits[i] |= bits;
            }
        }
    }
    output_bits
}

/// Project a bitvector slice for a given Thought Engine input,
/// applying bit masks and shifts as needed.
pub fn project_input(
    &self,
    full_state: &[u64],
    bit_mask: &[u64],
    shift: usize,
) -> Vec<u64> {
    full_state
        .iter()
        .zip(bit_mask.iter())
        .map(|(&state_part, &mask)| ((state_part & mask) >> shift))
        .collect()
}

/// Main update loop for one iteration
pub fn step(
    &mut self,
    engine_inputs: &[Vec<u64>],
    performance_feedback: f32,
    external_drive: f32,
) {
    // Update curiosity bit
    self.update_curiosity_bit(performance_feedback, external_drive);

    // For each Thought Engine, compute routed inputs and update state bits
    for i in 0..self.routing_matrix.len() {
        let routed_bits = self.route_and_integrate(engine_inputs, i);
        // Integrate routed_bits into self.state_bits[i] with chosen bitwise logic...
        // This could be as simple as OR, XOR, or more complex update functions
    }

    // Optionally, inject crawler engine output if curiosity_bit == true
    if self.curiosity_bit {
        // crawler_integration() => Update state_bits accordingly
    }
}
