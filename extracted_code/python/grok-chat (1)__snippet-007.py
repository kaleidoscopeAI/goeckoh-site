class Companion:
    def __init__(self):
        self.listener = Listener()
        self.heart = CrystallineHeart()
        self.voice = VoiceMimic()
        self.buffer = np.array([])

    def run(self):
        self.listener.start()
        while True:
            chunk = self.listener.get()
            if chunk is None: continue
            self.buffer = np.concatenate((self.buffer, chunk))
            if len(self.buffer) < CONFIG.sample_rate: continue  # Min length
            
            # VAD (simple energy-based fallback)
            rms = np.sqrt(np.mean(self.buffer**2))
            if rms < CONFIG.vad_threshold:
                self.buffer = self.buffer[-CONFIG.sample_rate // 2:]
                continue
            
            # Process
            stress = rms  # Simulated stress
            self.heart.anneal(stress)
            style = self.heart.get_emotion()
            
            # Transcribe (simulated—replace with whisper if installed)
            text = "echoed phrase"  # Placeholder for real STT
            
            synth = self.voice.synthesize(text, style, self.buffer)
            sd.play(synth, CONFIG.sample_rate)
            sd.wait()
            
            # Harvest
            self.voice.harvest(self.buffer, style)
            
            self.buffer = np.array([])

