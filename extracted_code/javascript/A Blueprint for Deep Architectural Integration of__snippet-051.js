let mut cog_state = CognitiveState::new();

// Initialize routing weights between 2 engines
cog_state.routing_weights.insert((0, 1), 0.5);
cog_state.routing_weights.insert((1, 2), 0.5);

let mut prev_uncertainty = cog_state.uncertainty;

for step in 0..100 {
    // Simulate uncertainty reduction
    cog_state.uncertainty = cog_state.compute_uncertainty();

    // Compute reward
    let reward = cog_state.compute_reward(prev_uncertainty, cog_state.uncertainty);

    // Update routing weights based on reward
    cog_state.update_routing_weights(reward);

    // Update curiosity tension
    cog_state.update_curiosity(reward);

    println!(
        "Step {}: Uncertainty {:.4}, Reward {:.4}, Curiosity tension {:.4}",
        step, cog_state.uncertainty, reward, cog_state.curiosity_tension
    );

    prev_uncertainty = cog_state.uncertainty;
}
