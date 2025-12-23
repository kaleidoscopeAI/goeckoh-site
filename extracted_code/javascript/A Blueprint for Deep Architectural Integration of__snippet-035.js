fn new(n_engines: usize) -> Self {
    let engines = (0..n_engines)
        .map(|_| Engine {
            params: vec![0.1; PARAM_SIZE],
            operator: Box::new(|input, params| {
                // Simple operator: elementwise multiply input by param vector mod length
                let mut output = Vec::new();
                for i in 0..input.len() {
                    let p = params[i % params.len()];
                    output.push(input[i] * p);
                }
                output
            }),
        })
        .collect();

    let routing = vec![vec![0.0; n_engines]; n_engines];

    Self {
        engines,
        routing,
        global_state: vec![0.5; GLOBAL_STATE_SIZE],
        threshold: 0.05,
        noise_level: 0.01,
        learning_rate_param: 0.001,
        learning_rate_route: 0.0001,
    }
}

// Compose inputs for engine i by concatenating global projection and routed outputs
fn compose_input(&self, i: usize) -> Vector {
    let mut input = Vec::new();

    // Project global state (here identity for demo)
    input.extend(self.global_state.iter());

    // Append outputs from connected engines passing threshold
    for j in 0..self.engines.len() {
        if self.routing[i][j] > self.threshold {
            let out = (self.engines[j].operator)(&self.global_state, &self.engines[j].params);
            input.extend(out.iter());
        }
    }

    input.into_iter().cloned().collect()
}

// Forward step to compute next global state
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
            *ns_val += val * self.routing[i][i]; // weight by self-route for simplification
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

// Dummy parameter update (e.g., gradient descent placeholder)
fn update_parameters(&mut self) {
    // For demo, just perturb params slightly
    for engine in &mut self.engines {
        for p in &mut engine.params {
            *p -= self.learning_rate_param * 0.01;
        }
    }
}

// Dummy routing update based on fake performance signal
fn update_routing(&mut self) {
    let n = self.engines.len();

    // Fake performance matrix with small random increments
    let mut rng = thread_rng();
    let normal = Normal::new(0.0, 0.001).unwrap();
    for i in 0..n {
        for j in 0..n {
            self.routing[i][j] += normal.sample(&mut rng);
            if self.routing[i][j] < 0.0 {
                self.routing[i][j] = 0.0;
            }
        }
        // Normalize row
        let sum: f64 = self.routing[i].iter().sum();
        if sum > 1.0 {
            for val in &mut self.routing[i] {
                *val /= sum;
            }
        }
    }
}
