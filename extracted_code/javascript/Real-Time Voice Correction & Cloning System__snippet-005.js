    pub fn new(asr_path: &str, tts_path: &str) -> PyResult<Self> {
        let env = Arc::new(Environment::builder().with_name("Goeckoh").build().unwrap());
        let asr = Session::builder(&env).unwrap().with_model_from_file(asr_path).unwrap();
        let tts = Session::builder(&env).unwrap().with_model_from_file(tts_path).unwrap();
        
        Ok(Self {
            asr_session: Arc::new(asr),
            tts_session: Arc::new(tts),
            input_buffer: Arc::new(Mutex::new(Vec::new())),
        })
    }

    pub fn start_hardware_stream(&self) -> PyResult<()> {
        let host = cpal::default_host();
        let device = host.default_input_device().expect("No input device");
        let config = device.default_input_config().unwrap();
        let buffer = Arc::clone(&self.input_buffer);

        let stream = device.build_input_stream(
            &config.into(),
            move |data: &[f32], _: &_| {
                let mut buf = buffer.lock().unwrap();
                buf.extend_from_slice(data);
            },
            |err| eprintln!("Audio Error: {}", err),
            None
        ).unwrap();

        stream.play().unwrap();
        std::mem::forget(stream); // Keep stream alive in background thread
        Ok(())
    }

    pub fn apply_spectral_gate(&self, mut audio: Vec<f32>) -> Vec<f32> {
        let n = audio.len();
        let mut planner = FftPlanner::new();
        let fft = planner.plan_fft_forward(n);
        let ifft = planner.plan_fft_inverse(n);

        let mut spectrum: Vec<Complex<f32>> = audio.iter().map(|&x| Complex::new(x, 0.0)).collect();
        fft.process(&mut spectrum);

        let noise_threshold = 0.015; // Dynamic floor placeholder
        for bin in spectrum.iter_mut() {
            if bin.norm_sqr() < noise_threshold {
                *bin = Complex::new(0.0, 0.0);
            }
        }

        ifft.process(&mut spectrum);
        spectrum.iter().map(|c| c.re / n as f32).collect()
    }
