fn run_simulation(system: &mut CognitiveSystem, inputs: Vec<Vec<f64>>, max_iters: usize) {
    for k in 0..max_iters {
        // Use external input at step k or zeros
        let input_k = inputs.get(k).unwrap_or(&vec![0.0; system.global_state.len()]);

        system.forward_step(input_k);

        // Check convergence, update parameters and routing here

        println!("Step {}: Global state = {:?}", k, system.global_state);
    }
}
