let mut system = CognitiveSystem::new(N_ENGINES);

// Initialize random routing for demo
for i in 0..N_ENGINES {
    for j in 0..N_ENGINES {
        system.routing[i][j] = if i == j { 1.0 } else { 0.0 };
    }
}

let external_input = vec![0.1; GLOBAL_STATE_SIZE];

for step in 0..100 {
    system.forward_step(&external_input);
    system.update_parameters();
    system.update_routing();

    println!("Step {}: Global state {:?}", step, &system.global_state[..5]); // print first 5 elements
}
