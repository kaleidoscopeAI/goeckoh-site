use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use ringbuf::{HeapRb, Consumer, Producer};
use std::sync::Arc;
use log::{info, error};

/// The capacity of the ring buffer (in samples).
/// 48000Hz * 0.1s = 4800 samples (100ms buffer).
/// We keep this tight to force low latency.
const LATENCY_BUFFER_SIZE: usize = 4096;

pub struct AudioStreamManager {
    // We must keep these streams alive, or audio stops immediately.
    _input_stream: cpal::Stream,
    _output_stream: cpal::Stream,
}

impl AudioStreamManager {
    pub fn new() -> Result<Self, anyhow::Error> {
        info!("AudioStreamManager: Initializing Host...");
        
        // 1. Get the Default Host (CoreAudio on iOS, Oboe/AAudio on Android)
        let host = cpal::default_host();

        // 2. Select Input (Mic) and Output (Speaker) Devices
        let input_device = host.default_input_device()
            .ok_or_else(|| anyhow::anyhow!("No input device found"))?;
        
        let output_device = host.default_output_device()
            .ok_or_else(|| anyhow::anyhow!("No output device found"))?;

        info!("Input Device: {}", input_device.name().unwrap_or("Unknown".into()));
        info!("Output Device: {}", output_device.name().unwrap_or("Unknown".into()));

        // 3. Configure Audio Format (Standardizing to 48kHz if possible)
        let config: cpal::StreamConfig = input_device.default_input_config()?.into();
        
        // 4. Create the Lock-Free Ring Buffer
        // This connects the Mic thread (Producer) to the Speaker thread (Consumer)
        // without ANY Mutexes or blocking.
        let ring = HeapRb::<f32>::new(LATENCY_BUFFER_SIZE);
        let (mut producer, mut consumer) = ring.split();

        // 5. Build the Input Stream (Microphone -> Ring Buffer)
        let input_stream = input_device.build_input_stream(
            &config,
            move |data: &[f32], _: &cpal::InputCallbackInfo| {
                // WRITE CALLBACK: High Priority
                // Simply push incoming mic data into the lock-free buffer.
                // If the buffer is full, we push_slice writes as much as possible.
                let _ = producer.push_slice(data);
            },
            move |err| error!("Input stream error: {}", err),
            None,
        )?;

        // 6. Build the Output Stream (Ring Buffer -> Speaker)
        // THIS is where the "Crystalline Heart" processing usually happens.
        let channels = config.channels as usize;
        let output_stream = output_device.build_output_stream(
            &config,
            move |data: &mut [f32], _: &cpal::OutputCallbackInfo| {
                // READ CALLBACK: Real-time Deadline
                process_audio_frame(data, &mut consumer, channels);
            },
            move |err| error!("Output stream error: {}", err),
            None,
        )?;

        // 7. Start the streams
        input_stream.play()?;
        output_stream.play()?;

        info!("Audio Pipeline Active. Latency optimized.");

        Ok(Self {
            _input_stream: input_stream,
            _output_stream: output_stream,
        })
    }
}

/// The critical hot-loop. This runs hundreds of times per second.
/// It MUST complete in under 2ms to avoid glitching.
fn process_audio_frame(
    output_buffer: &mut [f32], 
    consumer: &mut Consumer<f32, Arc<HeapRb<f32>>>,
    channels: usize
) {
    // Iterate over the output buffer and fill it with data from the Ring Buffer.
    for frame in output_buffer.chunks_mut(channels) {
        // Try to pop a sample from the Mic buffer
        let sample = match consumer.pop() {
            Some(s) => s,
            None => 0.0, // Buffer underflow (silence) - Silence is better than a crash
        };

        // TODO: INSERT CRYSTALLINE HEART MATH HERE
        // Right now, this is a "Passthrough" (Echo).
        // Future state: sample = ode_solver.process(sample);

        // Copy sample to all channels (Mono -> Stereo)
        for out_sample in frame.iter_mut() {
            *out_sample = sample;
        }
    }
}
