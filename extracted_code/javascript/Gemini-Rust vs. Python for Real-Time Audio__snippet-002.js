// core/src/audio/stream.rs
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use ringbuf::{HeapRb, Consumer, Producer};
use std::sync::{Arc, Mutex};
use log::{info, error};

use crate::crystalline_heart::CrystallineLattice;
use crate::neuro_acoustic::PsychoacousticProcessor;

const LATENCY_BUFFER_SIZE: usize = 4096;

pub struct AudioStreamManager {
    _input_stream: cpal::Stream,
    _output_stream: cpal::Stream,
}

impl AudioStreamManager {
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
}

/// The Hot Loop: Runs ~100x per second
fn process_audio_frame(
    output_buffer: &mut [f32],
    consumer: &mut Consumer<f32, Arc<HeapRb<f32>>>,
    lattice_handle: &Arc<Mutex<CrystallineLattice>>,
    dsp: &mut PsychoacousticProcessor,
    channels: usize
) {
    // 1. Try to lock the physics engine.
    // If the UI is reading it, we skip the physics update this frame (NO BLOCKING).
    if let Ok(mut lattice) = lattice_handle.try_lock() {
        
        // A. Read pending audio from Mic Buffer
        let mut temp_chunk = Vec::with_capacity(output_buffer.len() / channels);
        // (In production, use a stack array or pre-allocated buffer to avoid Vec alloc)
        
        // For simplicity: Process sample by sample (or small chunks)
        for frame in output_buffer.chunks_mut(channels) {
             let input_sample = consumer.pop().unwrap_or(0.0);
             
             // B. Physics Step (Simplified)
             // We inject energy based on the input sample amplitude
             let energy_packet = [input_sample.abs()]; 
             lattice.inject_energy(&energy_packet);
             
             // Advance physics by small time step (e.g. 0.001s)
             lattice.update(0.001);

             // C. Measure State for DSP
             let (valence, _, _) = lattice.measure_affective_state();
             
             // D. Apply DSP (Filter based on Valence)
             let processed_sample = dsp.process_output_frame(input_sample, valence);

             // E. Write to Output
             for out_sample in frame.iter_mut() {
                 *out_sample = processed_sample;
             }
        }
    } else {
        // FALLBACK: If we couldn't lock the lattice, just passthrough audio
        // to prevent silence/glitching.
        for frame in output_buffer.chunks_mut(channels) {
            let s = consumer.pop().unwrap_or(0.0);
            for out in frame { *out = s; }
        }
    }
}
