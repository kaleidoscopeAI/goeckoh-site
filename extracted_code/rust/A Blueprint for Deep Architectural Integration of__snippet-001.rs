let control = hf_controller.forward(global_state_vector);
system.config.temperature *= control.delta_T;
system.config.gamma = control.gamma;
