async fn new(voice_path: &str) -> Self {
    let piper = Piper::new().expect("Failed to create Piper");
    let voice = Voice::from_file(voice_path).expect("Failed to load voice");
    let sample_rate = voice.sample_rate();

    Self {
        piper,
        current_voice: voice,
        sample_rate,
    }
}

async fn synthesize_streaming(&self, text: &str) -> Vec<f32> {
    let mut synthesizer = self.piper
        .synthesize_streaming(&self.current_voice, text)
        .expect("Failed to synthesize");

    let mut audio_buffer = Vec::new();

    while let Some(audio_chunk) = synthesizer.next().await {
        let floats: Vec<f32> = audio_chunk
            .iter()
            .map(|&sample| sample as f32 / 32768.0)
            .collect();
        audio_buffer.extend(floats);
    }

    audio_buffer
}

fn clone_voice(&mut self, reference_audio: &[f32]) {
    // Note: Piper doesn't support voice cloning natively
    // This would require a custom model like F5-TTS compiled to Rust
    // For now, we load a pre-trained voice
    unimplemented!("Real-time voice cloning requires custom Rust bindings for F5-TTS/VITS");
}
