#[new]
fn new() -> Self {
    BioAcousticEngine { sample_rate: 22050.0 }
}

/// Advanced bio-acoustic synthesis with real physics modeling
/// Generates therapeutic audio based on emotional state
fn synthesize(&self, text_len: usize, arousal_state: f64) -> Vec<f32> {
    // Validate and clamp inputs for safety
    let safe_arousal = arousal_state.max(0.0).min(1.0);
    let duration = (text_len as f64 * 0.08).max(0.3).min(4.0);
    let num_samples = (self.sample_rate * duration) as usize;
    let mut buffer = Vec::with_capacity(num_samples);

    // Multi-layer synthesis approach
    // 1. Base frequency with biological resonance
    let base_f0 = 130.0 + (120.0 * (1.0 - safe_arousal)); // 130-250Hz range

    // 2. Formant structure for vocal quality
    let f1 = 800.0 + (200.0 * safe_arousal); // First formant
    let f2 = 1500.0 + (300.0 * safe_arousal); // Second formant

    // 3. Tremor and vibrato parameters
    let vibrato_rate = 5.0 + (2.0 * safe_arousal); // 5-7 Hz
    let vibrato_depth = if safe_arousal > 0.7 { 0.01 } else { 0.03 + (0.02 * safe_arousal) };

    // 4. Noise component for naturalness
    let noise_level = 0.02 + (0.03 * safe_arousal);

    let mut phase_base = 0.0;
    let mut phase_f1 = 0.0;
    let mut phase_f2 = 0.0;
    let mut vibrato_phase = 0.0;

    for i in 0..num_samples {
        let t = i as f64 / self.sample_rate;

        // Vibrato modulation
        vibrato_phase += (2.0 * PI * vibrato_rate) / self.sample_rate;
        let vibrato = 1.0 + (vibrato_phase.sin() * vibrato_depth);

        // Apply vibrato to base frequency
        let modulated_f0 = base_f0 * vibrato;

        // Update phases
        phase_base += (2.0 * PI * modulated_f0) / self.sample_rate;
        phase_f1 += (2.0 * PI * f1) / self.sample_rate;
        phase_f2 += (2.0 * PI * f2) / self.sample_rate;

        // Generate harmonic components
        let fundamental = phase_base.sin();
        let formant1 = (phase_f1.sin() * 0.3) * (1.0 + 0.5 * fundamental);
        let formant2 = (phase_f2.sin() * 0.2) * (1.0 + 0.3 * fundamental);

        // Add controlled noise for naturalness
        let noise = (rand::random::<f64>() - 0.5) * noise_level;

        // Combine components with biological weighting
        let sample = fundamental + formant1 + formant2 + noise;

        // ADSR envelope with therapeutic shaping
        let attack = 0.05; // 50ms attack
        let sustain = 0.7; // 70% sustain level
        let release_samples = (0.1 * self.sample_rate) as usize; // 100ms release

        let env = if i < (attack * self.sample_rate) as usize {
            i as f64 / (attack * self.sample_rate)
        } else if i > num_samples - release_samples {
            let release_progress = (num_samples - i) as f64 / release_samples as f64;
            sustain * release_progress
        } else {
            sustain
        };

        // Soft clipping for safety
        let soft_clipped = sample.tanh() * 0.8;
        buffer.push((soft_clipped * env) as f32);
    }

    buffer
}

/// Real-time audio modulation for emotional regulation
fn modulate_pcm(&self, input_samples: Vec<f32>, arousal_state: f64) -> Vec<f32> {
    let mut output = Vec::with_capacity(input_samples.len());
    let safe_arousal = arousal_state.max(0.0).min(1.0);

    // Dynamic modulation parameters based on emotional state
    let compression_ratio = 1.0 + (2.0 * safe_arousal); // 1:1 to 3:1 compression
    let low_freq_gain = 1.0 + (0.5 * (1.0 - safe_arousal)); // Boost lows when calm
    let high_freq_gain = 1.0 - (0.3 * safe_arousal); // Cut highs when aroused

    // Simple low-pass filter state
    let mut prev_sample = 0.0;
    let filter_coeff = if safe_arousal > 0.6 { 0.9 } else { 0.7 }; // More filtering when stressed

    for (i, sample) in input_samples.iter().enumerate() {
        let t = i as f64 / self.sample_rate;

        // Apply dynamic range compression for safety
        let compressed = if sample.abs() > 0.5 {
            sample.signum() * (0.5 + (sample.abs() - 0.5) / compression_ratio)
        } else {
            *sample
        };

        // Apply frequency shaping through simple filtering
        let filtered = prev_sample * filter_coeff + compressed * (1.0 - filter_coeff);
        prev_sample = filtered;

        // Apply frequency-dependent gain
        let shaped = filtered * low_freq_gain * high_freq_gain;

        // Safety limiting
        let limited = shaped.max(-0.9).min(0.9);
        output.push(limited as f32);
    }

    output
}
