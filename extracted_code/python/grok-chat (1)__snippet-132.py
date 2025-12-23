def __init__(self):
    self.engine = pyttsx3.init()  # Fallback TTS—mimic via rate/volume
    self.samples = {}  # Style: list of (f0, energy) prosody refs

def harvest(self, audio: np.ndarray, style: str):
    f0 = librosa.yin(audio, fmin=75, fmax=300)
    energy = librosa.feature.rms(y=audio)
    if style not in self.samples: self.samples[style] = []
    self.samples[style].append((f0, energy))

def synthesize(self, text: str, style: str, ref_audio: Optional[np.ndarray] = None) -> np.ndarray:
    if style in self.samples and self.samples[style]:
        f0_ref, energy_ref = np.random.choice(self.samples[style])
    else:
        f0_ref, energy_ref = None, None

    # Synthesize base (simulated mimicry)
    self.engine.setProperty('rate', 150 if style == "excited" else 100 if style == "calm" else 120)
    self.engine.setProperty('volume', 1.0)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        self.engine.save_to_file(text, tmp.name)
        self.engine.runAndWait()
        base_audio, _ = librosa.load(tmp.name, sr=CONFIG.sample_rate)
    os.unlink(tmp.name)

    # Apply prosody if ref
    if ref_audio is not None:
        f0, energy = librosa.yin(ref_audio, fmin=75, fmax=300), librosa.feature.rms(y=ref_audio)
        # Interpolate and apply (simplified)
        if len(base_audio) != len(energy[0]):
            energy = np.interp(np.linspace(0, 1, len(base_audio)), np.linspace(0, 1, len(energy[0])), energy[0])
        base_audio *= energy

    return base_audio

