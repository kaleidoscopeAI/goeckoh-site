class RealTimePipeline:
    def __init__(self, ref_audio: str = "assets/ref.wav"):
        self.recognizer = GLMRecognizer(model="glm-asr-nano-2512.onnx")  # Offline
        self.chatterbox = Chatterbox(model="chatterbox-turbo")  # Offline, zero-shot
        self.chatterbox.clone_voice(ref_audio)  # Zero-shot
        
        self.p = pyaudio.PyAudio()
        self.stream_in = self.p.open(format=pyaudio.paFloat32, channels=1, rate=16000, input=True, frames_per_buffer=160)
        self.stream_out = self.p.open(format=pyaudio.paFloat32, channels=1, rate=16000, output=True, frames_per_buffer=160)
        
        self.running = False
        logger.info("Improved pipeline: Chatterbox/GLM offline.")

    def correct_speech(self, text: str) -> str:
        start = time.time()
        text = re.sub(r'\b(you|he|she|they)\b', 'I', text, flags=re.IGNORECASE)
        text = re.sub(r'\b(want|needs)\b', 'want', text)
        text = re.sub(r'\b(is|are)\b', 'am', text)  # More grammar
        text = re.sub(r'\b(has|have)\b', 'have', text)
        lat = time.time() - start
        logger.info(f"Correction: {lat*1000:.2f} ms")
        return text

    def process_loop(self):
        while self.running:
            try:
                start = time.time()
                audio_chunk = np.frombuffer(self.stream_in.read(160), dtype=np.float32)
                
                text = self.recognizer.transcribe_stream(audio_chunk)  # Streaming
                
                if text:
                    corrected = self.correct_speech(text)
                    
                    audio_out = self.chatterbox.synthesize(corrected, emotion="neutral")  # Fast <150ms
                    
                    self.stream_out.write(audio_out.tobytes())
                    
                    lat = time.time() - start
                    logger.info(f"Latency: {lat*1000:.2f} ms")
            except Exception as e:
                logger.error(f"Error: {e}")

    def start(self):
        self.running = True
        threading.Thread(target=self.process_loop, daemon=True).start()

    def stop(self):
        self.running = False
        self.stream_in.stop_stream()
        self.stream_out.stop_stream()
        self.p.terminate()
