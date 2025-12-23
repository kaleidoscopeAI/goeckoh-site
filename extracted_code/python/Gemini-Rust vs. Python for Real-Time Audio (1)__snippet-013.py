// 1. Try to lock the physics engine.
// If the UI is reading it, we skip the physics update this frame (NO BLOCKING).
if let Ok(mut lattice) = lattice_handle.try_lock() {

    // A. Read pending audio from Mic Buffer
    let mut temp_chunk = Vec::with_capacity(output_buffer.len() / channels);
    // (In production, use a stack array or pre-allocated buffer to avoid Vec alloc)

    // For simplicity: Process sample by sample (or small chunks)
    for frame in output_buffer.chunks_mut(channels) {
         let input_sample = consumer.pop().unwrap_or(0.0);

         // B. Physics Step (Simplified)
         // We inject energy based on the input sample amplitude
         let energy_packet = [input_sample.abs()]; 
         lattice.inject_energy(&energy_packet);

         // Advance physics by small time step (e.g. 0.001s)
         lattice.update(0.001);

         // C. Measure State for DSP
         let (valence, _, _) = lattice.measure_affective_state();

         // D. Apply DSP (Filter based on Valence)
         let processed_sample = dsp.process_output_frame(input_sample, valence);

         // E. Write to Output
         for out_sample in frame.iter_mut() {
             *out_sample = processed_sample;
         }
    }
} else {
    // FALLBACK: If we couldn't lock the lattice, just passthrough audio
    // to prevent silence/glitching.
    for frame in output_buffer.chunks_mut(channels) {
        let s = consumer.pop().unwrap_or(0.0);
        for out in frame { *out = s; }
    }
}
