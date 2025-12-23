fn forward_step(&mut self, external_input: &Vector) {
    let n = self.engines.len();
    let mut outputs = vec![vec![]; n];

    // Compute outputs per engine
    for i in 0..n {
        let input_i = self.compose_input(i);
        outputs[i] = (self.engines[i].operator)(&input_i, &self.engines[i].params);
    }

    // Aggregate weighted outputs into new state
    let mut new_state = vec![0.0; self.global_state.len()];
    for i in 0..n {
        for (&val, ns_val) in outputs[i].iter().zip(new_state.iter_mut()) {
            *ns_val += val * self.routing[i][i]; // simplified weighting
        }
    }

    // Add external input
    for (ns_val, ext_val) in new_state.iter_mut().zip(external_input.iter()) {
        *ns_val += *ext_val;
    }

    // Add Gaussian noise
    let normal = Normal::new(0.0, self.noise_level).unwrap();
    let mut rng = thread_rng();
    for val in new_state.iter_mut() {
        *val += normal.sample(&mut rng);
    }

    self.global_state = new_state;
}

fn compose_input(&self, i: usize) -> Vector {
    let mut input = Vec::new();

    // Project global state (clip for example)
    input.extend(self.global_state.iter());

    // Append connected engine outputs (empty for benchmark simplicity)
    input
}
