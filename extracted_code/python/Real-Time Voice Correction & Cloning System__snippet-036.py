print("--- GOECKOH UNIVERSAL SYSTEM STARTING ---")

# 1. Initialize Hardware Kernel
# Loads the compact ASR model for offline use
kernel = goeckoh_kernel.UniversalKernel("assets/glm-asr-nano.onnx")
kernel.start_stream()
print("[Rust] Audio Kernel Active. Ring Buffer Ready.")

# 2. Initialize Voice Cloning
# Zero-shot clone in <150ms
tts = Chatterbox(model="chatterbox-turbo")
tts.clone_voice("assets/reference_voice.wav")
print("[AI] Voice Cloning Ready.")

# 3. Real-Time Loop
try:
    while True:
        # A. Fetch Audio from Rust (Non-blocking)
        # This pulls ~1024 samples at a time
        clean_audio = kernel.read_and_filter()

        if not clean_audio:
            time.sleep(0.005) # Yield if buffer empty
            continue

        # B. Transcribe (Hardware Accelerated)
        # Note: In a real loop, you accumulate chunks until VAD triggers
        text = kernel.transcribe(clean_audio)

        if text and text != "User input detected": # Filter empty/noise
            print(f"Input: {text}")

            # C. Deep Correction (ADK)
            # 'run' executes the SequentialAgent workflow [cite: 295]
            result = voice_coordinator.run(input=text)

            # Extract structured data [cite: 354]
            data = result.output.voice_data
            corrected = data.corrected_text
            tone = data.emotional_tone

            print(f"Output: {corrected} [{tone}]")

            # D. Synthesis
            # Generates audio in the user's cloned voice
            audio_out = tts.synthesize(corrected, emotion=tone)

            # Playback (Using simple sounddevice for Python-side output)
            # kernel.playback(audio_out) # Or pass back to Rust

except KeyboardInterrupt:
    print("System Shutdown.")

