// We modify 'new' to accept the shared Lattice
pub fn new(lattice_handle: Arc<Mutex<CrystallineLattice>>) -> Result<Self, anyhow::Error> {
    info!("AudioStreamManager: initializing...");

    let host = cpal::default_host();

    // Device selection (omitted error handling for brevity, same as before)
    let input_device = host.default_input_device().expect("No Input Device");
    let output_device = host.default_output_device().expect("No Output Device");
    let config: cpal::StreamConfig = input_device.default_input_config()?.into();

    // The Ring Buffer
    let ring = HeapRb::<f32>::new(LATENCY_BUFFER_SIZE);
    let (mut producer, mut consumer) = ring.split();

    // --- INPUT STREAM (Mic -> RingBuf) ---
    let input_stream = input_device.build_input_stream(
        &config,
        move |data: &[f32], _: &cpal::InputCallbackInfo| {
            let _ = producer.push_slice(data);
        },
        move |err| error!("Input error: {}", err),
        None,
    )?;

    // --- OUTPUT STREAM (RingBuf + Physics -> Speaker) ---
    // We clone the handle so the closure owns a reference to it
    let output_lattice = lattice_handle.clone();
    let channels = config.channels as usize;

    // Initialize our DSP processor
    let mut dsp = PsychoacousticProcessor::new();

    let output_stream = output_device.build_output_stream(
        &config,
        move |data: &mut [f32], _: &cpal::OutputCallbackInfo| {
            process_audio_frame(
                data, 
                &mut consumer, 
                &output_lattice, 
                &mut dsp,
                channels
            );
        },
        move |err| error!("Output error: {}", err),
        None,
    )?;

    input_stream.play()?;
    output_stream.play()?;

    Ok(Self {
        _input_stream: input_stream,
        _output_stream: output_stream,
    })
}
