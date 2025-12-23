use pyo3::prelude::*;
use std::f64::consts::PI;

#[pyclass]
struct BioEngine {
    sample_rate: f64,
}

#[pymethods]
impl BioEngine {
    #[new]
    fn new() -> Self {
        BioEngine { sample_rate: 22050.0 }
    }

    fn synthesize(&self, text_len: usize, arousal_state: f64) -> Vec<f32> {
        if text_len == 0 { return vec![0.0]; } // Safety check

        // Physics Logic
        let safe_arousal = arousal_state.clamp(0.0, 1.0);
        let duration = (text_len as f64 * 0.1).clamp(0.5, 5.0);
        let num_samples = (self.sample_rate * duration) as usize;
        
        let mut buffer = Vec::with_capacity(num_samples);
        let f0 = 150.0 - (safe_arousal * 20.0); 

        // Robotic damping during Meltdown (High stress)
        let jitter_intensity = if safe_arousal > 0.6 { 0.001 } else { 0.02 };
        
        let mut phase = 0.0;
        
        for i in 0..num_samples {
            let t = i as f64 / self.sample_rate;
            let jitter = (t * 30.0).sin() * jitter_intensity;
            let inst_f0 = f0 * (1.0 + jitter);
            
            phase += (2.0 * PI * inst_f0) / self.sample_rate;
            if phase > 2.0 * PI { phase -= 2.0 * PI; }

            // Soft Sawtooth (Rich Harmonic Content)
            let sample = 0.8 * phase.sin() - 0.2 * (2.0 * phase).sin();
            
            // Linear Attack/Decay Envelope
            let env = if i < 1000 { 
                i as f64 / 1000.0 
            } else if i > num_samples - 1000 { 
                (num_samples - i) as f64 / 1000.0 
            } else { 
                1.0 
            };

            buffer.push((sample * env) as f32);
        }
        
        buffer
    }

    fn modulate_pcm(&self, input_samples: Vec<f32>, arousal_state: f64) -> Vec<f32> {
        if input_samples.is_empty() { return vec![]; }

        let mut output = Vec::with_capacity(input_samples.len());
        let mut phase = 0.0;
        let safe_arousal = arousal_state.clamp(0.0, 1.0);

        // Tremor dynamics
        let tremolo_speed = if safe_arousal > 0.6 { 20.0 } else { 5.0 };
        let tremolo_depth = if safe_arousal > 0.8 { 0.0 } else { 0.15 * safe_arousal };

        for (i, &sample) in input_samples.iter().enumerate() {
            phase += tremolo_speed / self.sample_rate;
            if phase > 2.0 * PI { phase -= 2.0 * PI; }

            let amp_mod = 1.0 + (phase.sin() * tremolo_depth);
            
            // GCL Volume Compression (Safety Gating)
            let processed_sample = if safe_arousal > 0.8 {
                sample * 0.8 
            } else {
                sample
            };

            output.push(processed_sample * amp_mod as f32);
        }
        output
    }
}

#[pymodule]
fn bio_audio(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<BioEngine>()?;
    Ok(())
}