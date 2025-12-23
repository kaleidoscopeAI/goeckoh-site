#[new]
pub fn new(asr_path: &str) -> PyResult<Self> {
    let env = Arc::new(Environment::builder().build().unwrap());
    let asr = Session::builder(&env).unwrap().with_model_from_file(asr_path).unwrap();
    Ok(Self {
        asr_session: Arc::new(asr),
        input_buffer: Arc::new(Mutex::new(Vec::new())),
    })
}

pub fn process_stream(&self) -> PyResult<()> {
    let host = cpal::default_host();
    let device = host.default_input_device().expect("No device");
    let config = device.default_input_config().unwrap();
    let buffer = Arc::clone(&self.input_buffer);

    let stream = device.build_input_stream(
        &config.into(),
        move |data: &[f32], _: &_| {
            let mut buf = buffer.lock().unwrap();
            buf.extend_from_slice(data);
        },
        |e| eprintln!("{}", e),
        None
    ).unwrap();
    stream.play().unwrap();
    std::mem::forget(stream);
    Ok(())
}

pub fn apply_wiener_filter(&self, audio: Vec<f32>) -> Vec<f32> {
    let n = audio.len();
    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(n);
    let ifft = planner.plan_fft_inverse(n);
    let mut spec: Vec<Complex<f32>> = audio.iter().map(|&x| Complex::new(x, 0.0)).collect();
    fft.process(&mut spec);
    let noise_psd = 0.015;
    for bin in spec.iter_mut() {
        let gain = bin.norm_sqr() / (bin.norm_sqr() + noise_psd);
        *bin *= gain;
    }
    ifft.process(&mut spec);
    spec.iter().map(|c| c.re / n as f32).collect()
}
