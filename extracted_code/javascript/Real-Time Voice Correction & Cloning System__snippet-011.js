#[new]
pub fn new(model_path: &str) -> Self {
    let env = Environment::builder().build().unwrap();
    let asr_session = env.new_session(model_path).unwrap();
    Self { asr_session, vad: Arc::new(Mutex::new(Vec::new())) }
}

pub fn process_audio(&self, input: Vec<f32>) -> Vec<f32> {
    // Real-time Wiener Filter for Stationary Noise Reduction
    let n = input.len();
    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(n);
    let ifft = planner.plan_fft_inverse(n);

    let mut spectrum: Vec<Complex<f32>> = input.iter()
        .map(|&x| Complex::new(x, 0.0)).collect();
    fft.process(&mut spectrum);

    let noise_psd = 0.02; // Configurable noise floor
    for bin in spectrum.iter_mut() {
        let signal_psd = bin.norm_sqr();
        let gain = signal_psd / (signal_psd + noise_psd);
        *bin *= gain;
    }

    ifft.process(&mut spectrum);
    spectrum.iter().map(|c| c.re / n as f32).collect()
}
