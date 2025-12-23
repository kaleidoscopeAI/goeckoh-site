def __init__(self):
    self.config = Config()
    self.crystal = VoiceCrystal(self.config)
    self.lattice = EmotionalLattice()
    self.processor = SpeechProcessor()
    self.queue = queue.Queue()

def callback(self, indata, frames, time, status):
    if status:
        print(status)
    self.queue.put(indata.copy())

def run(self):
    print("Echo v4.0 Crystalline Heart – awake. Waiting for Jackson's voice...")
    with sd.InputStream(samplerate=self.config.sample_rate, channels=1, dtype='float32', callback=self.callback):
        while True:
            audio = self.queue.get()
            if np.abs(audio).mean() < 0.008:
                continue

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                sf.write(f.name, audio.flatten(), self.config.sample_rate)
                raw = self.processor.transcribe(f.name)
                os.unlink(f.name)

            if raw.strip():
                corrected = self.processor.correct(raw)
                self.lattice.update_from_audio(audio)
                style = self.lattice.get_style()
                wav = self.crystal.synthesize(corrected, style)
                sd.play(wav, samplerate=self.config.sample_rate)
                sd.wait()

def add_sample_from_mic(self, seconds=5, style="neutral"):
    print(f"Recording {style} sample for {seconds}s...")
    rec = sd.rec(int(seconds * self.config.sample_rate), samplerate=self.config.sample_rate, channels=1)
    sd.wait()
    self.crystal.add_sample(rec.flatten(), style)
    print("Sample saved to crystal.")

