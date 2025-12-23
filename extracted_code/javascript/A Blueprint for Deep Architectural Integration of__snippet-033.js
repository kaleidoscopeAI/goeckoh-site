fn new(n_engines: usize) -> Self {
    let engines = (0..n_engines)
        .map(|_| Engine {
            params: vec![0.0; PARAM_SIZE],    // Initialized to zero or random
            operator: Box::new(|x, p| vec![0.0; x.len()]), // Placeholder operator
        })
        .collect();

    let routing = vec![vec![0.0; n_engines]; n_engines]; // Zero initialization

    Self {
        engines,
        routing,
        global_state: vec![0.0; GLOBAL_STATE_SIZE],
        threshold: 0.1,
        noise_level: 0.01,
        learning_rate_param: 1e-3,
        learning_rate_route: 1e-4,
    }
}

// Project and concatenate inputs to engine i
fn compose_input(&self, i: usize) -> Vector {
    let mut input = Vec::new();

    // Append projection of global state for engine i (P_i[ S_k ]) - placeholder
    let proj_global = self.project_global_state(i);
    input.extend(proj_global);

    // Append outputs from engines j where R_{ij} > threshold
    for (j, weight) in self.routing[i].iter().enumerate() {
        if *weight > self.threshold {
            let output_j = self.engines[j].operator(&self.global_state, &self.engines[j].params);
            input.extend(output_j);
        }
    }

    input
}

// Placeholder for projection operator P_i on global state
fn project_global_state(&self, _i: usize) -> Vector {
    self.global_state.clone() // Identity projection for simplicity
}

// Forward evaluation - compute next global state S_{k+1}
fn forward_step(&mut self, external_input: &Vector) {
    let n = self.engines.len();
    let mut outputs = vec![vec![]; n];

    // Compute engine outputs
    for i in 0..n {
        let input_i = self.compose_input(i);
        outputs[i] = (self.engines[i].operator)(&input_i, &self.engines[i].params);
    }

    // Aggregate outputs with routing and external input to form new state
    let mut new_state = vec![0.0; self.global_state.len()];

    for i in 0..n {
        let weighted_output: Vector = outputs[i]
            .iter()
            .map(|val| val * self.routing[i][i])  // Weight with self-route for simplicity
            .collect();
        new_state.iter_mut()
            .zip(weighted_output.iter())
            .for_each(|(ns, wo)| *ns += *wo);
    }

    // Add projected external input
    new_state.iter_mut()
        .zip(external_input.iter())
        .for_each(|(ns, ei)| *ns += *ei);

    // Add Gaussian noise
    for val in &mut new_state {
        *val += self.noise_level * random_gaussian();
    }

    self.global_state = new_state;
}

// Gradient descent parameter update per engine (pseudo)
fn update_parameters(&mut self, gradients: &Vec<Vector>) {
    for (i, grad) in gradients.iter().enumerate() {
        for (p, g) in self.engines[i].params.iter_mut().zip(grad.iter()) {
            *p -= self.learning_rate_param * g;
        }
    }
}

// Routing weights update per edge (pseudo)
fn update_routing(&mut self, performance_matrix: &Matrix) {
    let n = self.engines.len();

    for i in 0..n {
        for j in 0..n {
            self.routing[i][j] += self.learning_rate_route * performance_matrix[i][j];
            // Enforce non-negativity
            if self.routing[i][j] < 0.0 {
                self.routing[i][j] = 0.0;
            }
        }
        // Normalize row to sum to ≤ 1
        let row_sum: f64 = self.routing[i].iter().sum();
        if row_sum > 1.0 {
            for val in &mut self.routing[i] {
                *val /= row_sum;
            }
        }
    }
}
