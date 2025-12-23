def __init__(self):
    print("[System] Initializing Echo Prime...")
    self.heart = CrystallineHeart()
    self.aba = AbaEngine()
    self.voice = VoiceCrystal()
    self.subconscious = OrganicSubconscious()

    # Start Subconscious
    self.sub_thread = threading.Thread(target=self.subconscious.run_background, daemon=True)
    self.sub_thread.start()

    # Audio Setup
    self.q = queue.Queue()
    try:
        from faster_whisper import WhisperModel
        self.whisper = WhisperModel("tiny.en", device="cpu", compute_type="int8")
        print("[System] Ears (Whisper) active.")
    except ImportError:
        print("[System] Faster-Whisper not found. Speech recognition will be simulated text-only.")
        self.whisper = None

def listen_loop(self):
    """Real-time audio processing loop."""
    print("[System] Listening... (Press Ctrl+C to stop)")

    # VAD / Audio Callback
    def callback(indata, frames, time, status):
        self.q.put(indata.copy())

    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, callback=callback):
        buffer = []
        silence_frames = 0

        while True:
            try:
                audio_chunk = self.q.get()
                rms = np.sqrt(np.mean(audio_chunk**2))

                # Feed the Organic Subconscious directly
                self.subconscious.feed(rms * 100) # Amplify small signals

                if rms > 0.01: # Speech detected
                    buffer.append(audio_chunk)
                    silence_frames = 0
                else:
                    silence_frames += 1

                # End of utterance detection (approx 1.2s silence)
                if silence_frames > 20 and buffer: 
                    audio_full = np.concatenate(buffer)
                    self.process_utterance(audio_full)
                    buffer = [] # Reset

            except KeyboardInterrupt:
                break
            except Exception as e:
                pass # Keep running

def process_utterance(self, audio_data):
    """The Cognitive Pipeline."""
    # 1. Transcribe
    text = ""
    if self.whisper:
        # Convert to float32 for Whisper
        audio_float = audio_data.flatten().astype(np.float32)
        segments, _ = self.whisper.transcribe(audio_float, beam_size=5)
        text = " ".join([s.text for s in segments])

    if not text.strip(): return

    print(f"\n🎤 User: {text}")

    # 2. Update Heart
    rms = np.sqrt(np.mean(audio_data**2))
    arousal, valence = self.heart.update(rms)

    # 3. ABA Check
    intervention = self.aba.evaluate(arousal, text)

    if intervention:
        # High priority intervention
        self.voice.speak(intervention, style="calm")
    else:
        # 4. LLM Response (The "Self")
        response = self.query_llm(text, arousal)
        self.voice.speak(response, style="neutral")

def query_llm(self, text, arousal):
    """Get a response from the local LLM personality."""
    try:
        import ollama
        prompt = f"""
        You are Echo, a supportive inner voice for an autistic child.
        The child just said: "{text}".
        Their arousal level is {arousal:.2f} (0-10).
        Reply in 1 short sentence. Be kind, validating, and use 'I' statements as if you are their inner voice.
        """
        res = ollama.generate(model=OLLAMA_MODEL, prompt=prompt)
        return res['response']
    except:
        return f"I hear you saying {text}."

