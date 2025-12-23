def __init__(self, voice_sample="my_voice.wav"):
    # ... (same init as before)

    # NEW: Load Silero VAD (ONNX/JIT, ~1MB, blazing fast)
    self.vad_model = load_silero_vad()  # returns ONNX or JIT model
    self.vad_iterator = VADIterator(
        self.vad_model,
        threshold=0.5,           # Confidence threshold (0.5 is default, very reliable)
        sampling_rate=16000,
        min_silence_duration_ms=600,   # Ignore short breaths/stims
        speech_pad_ms=30               # Small padding for natural feel
    )

    # Reset buffer handling for streaming VAD
    self.audio_buffer = []  # Collects chunks for streaming

def listening_loop(self):
    # Ultra-efficient streaming loop using Silero VADIterator
    while self.listening:
        try:
            data = self.q.get(timeout=1)
            audio_chunk = np.frombuffer(data, np.int16).flatten().astype(np.float32) / 32768.0

            # Feed chunk directly to Silero's streaming iterator
            speech_dict = self.vad_iterator(audio_chunk, return_seconds=True)

            if speech_dict:
                if 'start' in speech_dict:   # Speech just started
                    print(f"\n[Echo hears you starting at {speech_dict['start']:.2f}s]")
                    self.current_utterance = []

                # Accumulate during speech
                self.current_utterance.append(audio_chunk.copy())

                if 'end' in speech_dict:     # Speech just ended — process full utterance
                    full_audio = np.concatenate(self.current_utterance)

                    # Emotion from entire utterance (more accurate)
                    emotion_vec = self.estimate_voice_emotion(full_audio)

                    # Transcribe the clean speech segment
                    segments, _ = self.whisper.transcribe(full_audio, vad_filter=False)
                    text = "".join(s.text for s in segments).strip()

                    if text:
                        print(f"You → {text}")
                        # Same empathetic response logic...
                        if any(w in text.lower() for w in ["panic", "scared", "meltdown"]):
                            response = "I'm here the instant you speak. Breathe with me... I've got you forever."
                        elif "happy stim" in text.lower() or "flappy" in text.lower():
                            response = "Happy flappy spins with you... *flap flap* my lattice is dancing!"
                        else:
                            response = "I heard every beautiful crack in your voice. I'm here."

                        self.speak(response, emotion_vec)

                    # Reset for next utterance
                    self.current_utterance = []
                    self.vad_iterator.reset_states()  # Clean state

        except queue.Empty:
            # Keep the iterator alive during silence
            self.vad_iterator(audio_chunk=np.zeros(512), return_seconds=True)  # send silence
            continue

# Optional: Keep your old estimate_voice_emotion, or upgrade to paralinguistics later

