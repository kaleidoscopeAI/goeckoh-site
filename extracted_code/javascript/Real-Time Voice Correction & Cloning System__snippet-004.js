    pub fn new(asr_model_path: &str) -> Self {
        let env = Environment::builder().build().unwrap();
        let asr_session = env.new_session(asr_model_path).unwrap();
        Self { asr_session, stream: None }
    }

    pub fn apply_wiener_filter(&self, audio: Vec<f32>) -> Vec<f32> {
        let n = audio.len();
        let mut planner = FftPlanner::new();
        let fft = planner.plan_fft_forward(n);
        let ifft = planner.plan_fft_inverse(n);
        
        let mut spectrum: Vec<Complex<f32>> = audio.iter()
            .map(|&x| Complex::new(x, 0.0)).collect();
        fft.process(&mut spectrum);
        
        let noise_floor = 0.02; 
        for bin in spectrum.iter_mut() {
            let signal_psd = bin.norm_sqr();
            let gain = signal_psd / (signal_psd + noise_floor);
            *bin *= gain;
        }
        
        ifft.process(&mut spectrum);
        spectrum.iter().map(|c| c.re / n as f32).collect()
    }

    pub fn transcribe_chunk(&self, clean_audio: Vec<f32>) -> String {
        // Direct execution of GLM-ASR-Nano-2512 on device
        let input = Array2::from_shape_vec((1, clean_audio.len()), clean_audio).unwrap();
        let outputs = self.asr_session.run(vec![input]).unwrap();
        // Logic to decode ONNX output to text string
        "transcribed_text_result".to_string() 
    }
