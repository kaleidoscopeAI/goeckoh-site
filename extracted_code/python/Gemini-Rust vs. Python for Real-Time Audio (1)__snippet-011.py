// Iterate over the output buffer and fill it with data from the Ring Buffer.
for frame in output_buffer.chunks_mut(channels) {
    // Try to pop a sample from the Mic buffer
    let sample = match consumer.pop() {
        Some(s) => s,
        None => 0.0, // Buffer underflow (silence) - Silence is better than a crash
    };

    // TODO: INSERT CRYSTALLINE HEART MATH HERE
    // Right now, this is a "Passthrough" (Echo).
    // Future state: sample = ode_solver.process(sample);

    // Copy sample to all channels (Mono -> Stereo)
    for out_sample in frame.iter_mut() {
        *out_sample = sample;
    }
}
