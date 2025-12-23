class AudioSystem:
    def __init__(self):
        self.q = queue.Queue()
        self.running = True
        self.bio_engine = bio_audio.BioEngine() if RUST else None
        self.neural_engine = neural_speech.VoiceEngine() if NEURAL else None
        threading.Thread(target=self._loop, daemon=True).start()

    def enqueue_response(self, text: str, arousal: float):
        pcm = None
        if NEURAL and self.neural_engine.use_neural and RUST:
             # Real Cloning path would rely on assets/ref_voice.wav existance check here
             # defaulting to mock/synth if ref not found
             pass 
        
        if pcm is None and RUST:
            pcm = np.array(self.bio_engine.synthesize(len(text), arousal), dtype=np.float32)
        
        if pcm is not None: self.q.put(pcm)

    def _loop(self):
        while self.running:
            if not AUDIO: time.sleep(1); continue
            try:
                data = self.q.get(timeout=1)
                sd.play(data, samplerate=22050, blocking=True)
            except: continue

