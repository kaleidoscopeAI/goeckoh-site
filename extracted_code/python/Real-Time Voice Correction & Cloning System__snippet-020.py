import sys
import time
import numpy as np
import goeckoh_kernel  # Our compiled Rust module
from goeckoh_brain.agents import voice_brain
from chatterbox import Chatterbox  # SOTA Zero-shot Cloning Library

def main():
    print("Initializing Goeckoh Universal System...")
    
    # 1. Initialize Rust Engine (Hardware Level)
    engine = goeckoh_kernel.AudioEngine("assets/glm-asr-nano.onnx")
    engine.start_stream()
    
    # 2. Initialize Voice Cloning (The "Voice")
    # This loads the user's reference voice ONCE into memory
    cloner = Chatterbox(model="chatterbox-turbo")
    cloner.clone_voice("assets/reference_voice.wav")
    
    print("System Active. Listening...")

    # 3. Main Event Loop (The Heartbeat)
    buffer = []
    try:
        while True:
            # Simulate fetching a 100ms chunk from the Rust ringbuffer
            # In production, you expose a 'read_buffer' method in Rust
            raw_audio = [0.0] * 1600 # Placeholder for 100ms at 16khz
            
            # A. Signal Processing (Rust)
            clean_audio = engine.process_audio_frame(raw_audio)
            
            # B. Transcription (Rust + ONNX)
            # Only transcribe if VAD (Voice Activity Detection) triggers
            text = engine.transcribe(clean_audio)
            
            if text:
                print(f"Heard: {text}")
                
                # C. Deep Understanding (Python/ADK)
                # This runs the Agent logic defined in 'agents.py'
                brain_output = voice_brain.run(input=text)
                
                corrected_text = brain_output.output.corrected_text
                intent = brain_output.output.intent_tag
                print(f"Corrected: {corrected_text} [{intent}]")
                
                # D. Synthesis (Zero-Shot Cloning)
                audio_out = cloner.synthesize(corrected_text, emotion=intent)
                
                # Playback is handled by the Rust Output Stream implicitly
                # or we push back to the buffer:
                # engine.push_to_speakers(audio_out)

            time.sleep(0.01) # Yield to prevent CPU hogging

    except KeyboardInterrupt:
        print("Shutting down system.")

