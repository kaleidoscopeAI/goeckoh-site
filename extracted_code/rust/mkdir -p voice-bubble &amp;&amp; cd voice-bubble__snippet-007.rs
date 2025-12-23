use ort::{Session, Value};

struct SileroVAD {
    session: Session,
    threshold: f32,
    sample_rate: u32,
}

impl SileroVAD {
    fn new(model_path: &str) -> Self {
        let session = Session::builder()
            .unwrap()
            .with_model_from_file(model_path)
            .unwrap();
        
        Self {
            session,
            threshold: 0.5,
            sample_rate: 16000,
        }
    }
    
    fn is_speech(&self, audio: &[f32]) -> bool {
        // Prepare input tensor: [1, audio_len]
        let input_shape = vec![1, audio.len() as i64];
        let input_array = ort::ndarray::Array2::from_shape_vec(
            (1, audio.len()),
            audio.to_vec()
        ).unwrap();
        
        let inputs = vec![Value::from_array(self.session.allocator(), &input_array).unwrap()];
        
        // Run inference
        let outputs = self.session.run(inputs).unwrap();
        let output = outputs[0].try_extract_tensor::<f32>().unwrap();
        let confidence = output.view()[[0, 0]];
        
        confidence > self.threshold
    }
}
