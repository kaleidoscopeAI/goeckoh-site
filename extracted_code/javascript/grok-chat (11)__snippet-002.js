#[frb(sync)]
pub fn new() -> Self {
    let env = Environment::builder().build().unwrap();
    let asr_session = env.new_session("assets/glm-asr-nano-2512.onnx").unwrap();
    let tts_session = env.new_session("assets/fun-cosyvoice-3.onnx").unwrap();
    let vad = Vad::new();
    let resampler = FftFixedInOut::<f32>::new(44100, 16000, 1024, 1, 1).unwrap();
    let fft_planner = FftPlanner::new();

    Self { env, asr_session, tts_session, vad, resampler: Arc::new(Mutex::new(resampler)), stream: None, fft_planner }
}

#[frb]
pub fn start_stream(&mut self) -> Result<(), String> {
    let host = cpal::default_host();
    let input = host.default_input_device().ok_or("No input")?;
    let config = input.default_input_config().unwrap();

    let data = Arc::new(Mutex::new(Vec::new()));
    let data_clone = data.clone();

    let stream = input.build_input_stream(
        &config.into(),
        move |d: &[f32], _: &_| {
            let mut buf = data_clone.lock().unwrap();
            buf.extend_from_slice(d);
        },
        |err| eprintln!("Error: {err}"),
        None,
    ).unwrap();

    stream.play().unwrap();
    self.stream = Some(stream);
    Ok(())
}

#[frb]
pub fn process_audio(&mut self, ref_audio: Option<String>) -> String {
    let mut buf = self.resampler.lock().unwrap();
    let data = Arc::new(Mutex::new(Vec::new()));

    // VAD
    let voiced = self.vad.process(&data.lock().unwrap());

    // Resample
    let resampled = buf.process(&[voiced.clone()]).unwrap()[0].clone();

    // ASR with GLM-Nano
    let input = Array2::from_shape_vec((1, resampled.len()), resampled).unwrap();
    let outputs = self.asr_session.run(vec![input]).unwrap();
    let text = outputs[0].index_axis(0, 0).to_vec().iter().map(|&i| i.to_string()).collect::<String>();  // Sim text extract

    // Correction
    let corrected = text.replace("you", "I").replace("want", "want").replace("needs", "want");

    // TTS/Cloning with Fun-CosyVoice
    let tts_input = Array::from_vec(corrected.as_bytes().to_vec());
    let audio_out = self.tts_session.run(vec![tts_input]).unwrap()[0].to_vec();  // Raw audio

    // Noise reduction (Wiener in Rust)
    let clean = self.wiener_filter(&audio_out);

    // Clone if ref
    if let Some(ref_path) = ref_audio {
        // Zero-shot cloning stub (integrate CosyVoice cloning)
    }

    "Processed: ".to_owned() + &corrected
}

fn wiener_filter(&self, audio: &Vec<f32>) -> Vec<f32> {
    let n = audio.len();
    let mut fft = self.fft_planner.plan_fft_forward(n);
    let mut ifft = self.fft_planner.plan_fft_inverse(n);

    let mut spectrum = audio.clone().into_iter().map(|x| rustfft::num_complex::Complex::new(x, 0.0)).collect::<Vec<_>>();
    fft.process(&mut spectrum);

    // PSD/noise est (simplified)
    let noise_psd = 0.01;  // Est from silence
    let signal_psd = spectrum.iter().map(|c| c.norm_sqr()).collect::<Vec<_>>();
    let gain = signal_psd.iter().map(|&s| s / (s + noise_psd)).collect::<Vec<_>>();

    for (s, g) in spectrum.iter_mut().zip(gain) {
        *s *= g;
    }

    ifft.process(&mut spectrum);
    spectrum.into_iter().map(|c| c.re / n as f32).collect()
}
