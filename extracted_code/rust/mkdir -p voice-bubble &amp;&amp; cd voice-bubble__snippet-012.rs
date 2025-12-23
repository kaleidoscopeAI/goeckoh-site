// echo_core.rs - your exact main.rs
use cpal::traits::*;
use ringbuf::{HeapRb, Producer, Consumer};
use whisper_rs::{Context, FullParams, SamplingStrategy};
use std::sync::Arc;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let (_prod, cons) = HeapRb::<f32>::new(16384).split();
    let audio_rx = Arc::new(cons);
    
    // CPAL input → ringbuf
    let host = cpal::default_host();
    let device = host.default_input_device()?;
    let config = device.default_input_config()?;
    
    let stream = device.build_input_stream(
        &config.into(),
        move |data: &[f32], _: &cpal::InputCallbackInfo| {
            // VAD simple threshold
            let rms: f32 = data.iter().map(|&x| x*x).sum::<f32>().sqrt() / data.len() as f32;
            if rms > 0.02 {
                let mut slice = audio_rx.slice();
                let len = slice.len().min(data.len());
                slice[0..len].copy_from_slice(&data[0..len]);
                audio_rx.advance_write(len);
            }
        },
        |err| eprintln!("Audio err: {}", err),
        None,
    )?;
    
    stream.play()?;
    
    // Whisper streaming loop
    let ctx = Context::new("whisper-tiny.en.bin")?;
    let mut state = ctx.create_state()?;
    let mut params = FullParams::new(SamplingStrategy::Greedy{ best_of: 1 });
    
    loop {
        if let Some(audio) = audio_rx.pop_slice(16000) {  // 1s chunks
            // Whisper process → text
            let text = process_whisper(&mut state, &mut params, &audio)?;
            
            if !text.is_empty() {
                // Piper TTS → audio queue
                let tts_audio = piper_synthesize(&text).await?;
                // Push to output ringbuf (play immediately)
            }
        }
        tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;
    }
}
