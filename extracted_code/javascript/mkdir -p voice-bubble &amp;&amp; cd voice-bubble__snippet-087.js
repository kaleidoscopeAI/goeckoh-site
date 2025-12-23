async fn new() -> Result<Self> {
    let host = cpal::default_host();
    let input_device = host.default_input_device().unwrap();
    let output_device = host.default_output_device().unwrap();

    let config = StreamConfig {
        channels: 1,
        sample_rate: cpal::SampleRate(16000),
        buffer_size: BufferSize::Fixed(512), // 32ms frames
    };

    // Input callback
    let input_queue = Arc::new(SegQueue::new());
    let input_queue_clone = input_queue.clone();

    let input_stream = input_device.build_input_stream(
        &config,
        move |data: &[f32], _: &cpal::InputCallbackInfo| {
            let frame = AudioFrame::from_slice(data);
            input_queue_clone.push(frame);
        },
        move |err| eprintln!("Input error: {}", err),
        None,
    )?;

    // Output callback
    let output_buffer = HeapRb::<f32>::new(4096);
    let (mut prod, cons) = output_buffer.split();
    let output_queue = Arc::new(cons);

    let output_stream = output_device.build_output_stream(
        &config,
        move |data: &mut [f32], _: &cpal::OutputCallbackInfo| {
            for sample in data.iter_mut() {
                *sample = prod.pop().unwrap_or(0.0);
            }
        },
        move |err| eprintln!("Output error: {}", err),
        None,
    )?;

    Ok(Self {
        input_stream,
        output_stream,
        input_queue,
        output_queue,
    })
}
