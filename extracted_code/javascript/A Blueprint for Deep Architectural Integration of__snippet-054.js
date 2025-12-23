pub fn update_curiosity_bit(&mut self, perf: f32, external_drive: f32) {
    let input = self.rho * if self.curiosity_bit { self.lambda } else { 0.0 }
        - self.sigma * perf
        + self.beta * external_drive;
    self.curiosity_bit = input > self.theta;
}

```
pub fn route_and_integrate(&self, engine_inputs: &[Vec<u64>], engine_idx: usize) -> Vec<u64> {
```
    let mut out = vec![0u64; engine_inputs[^57_0].len()];
    for (src_idx, input) in engine_inputs.iter().enumerate() {
        if self.routing_matrix[engine_idx][src_idx] {
            for (i, &bits) in input.iter().enumerate() {
                out[i] |= bits;
            }
        }
    }
    out
}

pub fn project(&self, full_state: &[u64], mask: &[u64], shift: usize) -> Vec<u64> {
    full_state.iter()
        .zip(mask)
        .map(|(&state, &m)| ((state & m) >> shift))
        .collect()
}

pub fn step(&mut self, engine_inputs: &[Vec<u64>], perf: f32, ext_drive: f32) {
    self.update_curiosity_bit(perf, ext_drive);
    for i in 0..self.routing_matrix.len() {
        let routed = self.route_and_integrate(engine_inputs, i);
        // integrate routed into state_bits[i] using bitwise operation...
    }
    if self.curiosity_bit {
        // Activate crawler engine and integrate output
    }
}
