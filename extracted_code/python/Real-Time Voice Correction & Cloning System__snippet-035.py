// We hold the stream to keep it alive
_stream: Option<cpal::Stream>,
// Consumer reads audio from the hardware thread
consumer: Arc<Mutex<Consumer<f32>>>,
// FFT planner for the spectral gate
fft_planner: Arc<Mutex<FftPlanner<f32>>>,
// ONNX Session for ASR
asr_session: Session,
