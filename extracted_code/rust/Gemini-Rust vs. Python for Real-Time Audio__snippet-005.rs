// core/src/neuro_acoustic/mod.rs

/// A simple implementation of a Neuro-Acoustic filter.
/// It measures signal energy (RMS) and applies a low-pass filter
/// modulated by the "Valence" of the lattice.
pub struct PsychoacousticProcessor {
    // Simple state for a low-pass filter
    prev_sample: f32,
}

impl PsychoacousticProcessor {
    pub fn new() -> Self {
        Self { prev_sample: 0.0 }
    }

    /// Calculates the Root Mean Square (Energy) of a buffer chunk.
    /// This tells the Lattice how "loud" the input is.
    pub fn calculate_energy(&self, buffer: &[f32]) -> f32 {
        if buffer.is_empty() {
            return 0.0;
        }
        let sum_squares: f32 = buffer.iter().map(|&x| x * x).sum();
        (sum_squares / buffer.len() as f32).sqrt()
    }

    /// Modulates the audio output based on emotional state.
    /// If Valence is LOW (Stress), we filter high frequencies (muffle).
    /// If Valence is HIGH (Calm), we let the voice sound clear.
    pub fn process_output_frame(&mut self, sample: f32, valence: f32) -> f32 {
        // Map Valence (0.0 - 1.0) to a Filter Coefficient (Alpha)
        // 0.0 (Stressed) -> 0.1 (Heavy Filtering)
        // 1.0 (Happy)    -> 0.9 (Light Filtering)
        let alpha = 0.1 + (valence * 0.8);
        
        // Basic One-Pole Low-Pass Filter
        let output = (alpha * sample) + ((1.0 - alpha) * self.prev_sample);
        self.prev_sample = output;
        
        output
    }
}
