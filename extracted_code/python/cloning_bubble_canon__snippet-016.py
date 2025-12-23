real_features = analyze_attempt(child_audio, sr)
synth_features = controls_to_attempt_features(controls)
error = compute_psychoacoustic_distance(real_features, synth_features)
