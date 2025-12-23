base_dir: Path
max_samples_per_style: int = 32
samples: Dict[Style, List[VoiceSample]] = field(default_factory=lambda: {"neutral": [], "calm": [], "excited": []})

def __post_init__(self):
    self.base_dir.mkdir(parents=True, exist_ok=True)
    self.load_existing()

def add_sample_from_wav(self, wav: np.ndarray, style: Style, name: Optional[str] = None):
    dir_path = self.base_dir / style
    dir_path.mkdir(exist_ok=True)
    path = dir_path / f"{uuid.uuid4()}.wav"
    sf.write(path, wav, 16000)
    quality_score = self._assess_quality(wav)
    sample = VoiceSample(path, len(wav)/16000, np.sqrt(np.mean(wav**2)), style, quality_score, time.time())
    self.samples[style].append(sample)
    self.samples[style].sort(key=lambda x: (x.quality_score, x.added_ts), reverse=True)
    self.samples[style] = self.samples[style][:self.max_samples_per_style]
    return path

def get_best_sample(self, style: Style) -> Optional[np.ndarray]:
    if not self.samples[style]:
        return None
    best = self.samples[style][0]
    wav, _ = sf.read(best.path)
    return np.mean(wav, axis=1) if wav.ndim > 1 else wav

def _assess_quality(self, wav): 
    # Simple clarity, low noise, good energy
    return float(np.mean(librosa.feature.rms(y=wav))) * 100

