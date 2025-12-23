"""Manages the collection of the user's voice samples (the "Voice Crystal")."""

audio_cfg: AudioSettings
paths: PathRegistry
samples: list[VoiceSample] = field(default_factory=list)
max_samples: int = 50

def __post_init__(self):
    self.paths.voices_dir.mkdir(parents=True, exist_ok=True)
    self.load_existing()

def load_existing(self):
    """Load all .wav files from the voice directory."""
    for wav_path in sorted(self.paths.voices_dir.glob("**/*.wav")):
        try:
            data, sr = sf.read(wav_path, dtype="float32")
            if sr != self.audio_cfg.sample_rate:
                data = librosa.resample(
                    y=data, orig_sr=sr, target_sr=self.audio_cfg.sample_rate
                )

            duration = len(data) / float(self.audio_cfg.sample_rate)
            rms = (
                float(np.sqrt(np.mean(np.square(data)))) if data.size > 0 else 0.0
            )
            # Quality score can be stored in filename or a metadata file in future
            self.samples.append(VoiceSample(wav_path, duration, rms, 1.0))
        except Exception as e:
            print(f"Failed to load voice sample {wav_path}: {e}")

def add_sample(self, wav: np.ndarray, quality_score: float) -> Path | None:
    """Adds a new voice sample to the profile if quality is high enough."""
    if quality_score < 0.9:
        return None

    path = self.paths.voices_dir / f"sample_{int(random.random() * 1e8)}.wav"
    sf.write(path, wav, self.audio_cfg.sample_rate)

    duration = len(wav) / float(self.audio_cfg.sample_rate)
    rms = float(np.sqrt(np.mean(np.square(wav))))
    self.samples.append(VoiceSample(path, duration, rms, quality_score))

    self._prune()
    return path

def _prune(self):
    """Keeps only the highest quality samples if count exceeds max_samples."""
    if len(self.samples) > self.max_samples:
        self.samples.sort(key=lambda s: s.quality_score, reverse=True)
        self.samples = self.samples[: self.max_samples]

def pick_reference(self) -> Path | None:
    """Picks a random, high-quality sample to use for voice cloning."""
    if not self.samples:
        return None
    # Prefer higher quality samples
    high_quality_samples = [s for s in self.samples if s.quality_score > 0.95]
    if high_quality_samples:
        return random.choice(high_quality_samples).path
    return random.choice(self.samples).path

def maybe_adapt_from_attempt(
    self,
    attempt_wav: np.ndarray,
    style: Literal["neutral", "calm", "excited"] = "neutral",
    quality_score: float = 0.0,
    min_quality: float = 0.8,
) -> Path | None:
    """Add a new facet when an attempt is strong enough."""
    if quality_score < min_quality:
        return None
    style_dir = self.paths.voices_dir / style
    style_dir.mkdir(parents=True, exist_ok=True)
    path = style_dir / f"{style}_{uuid.uuid4().hex[:8]}.wav"
    sf.write(path, attempt_wav, self.audio_cfg.sample_rate)
    duration = len(attempt_wav) / float(self.audio_cfg.sample_rate)
    rms = float(np.sqrt(np.mean(np.square(attempt_wav)) + 1e-8))
    self.samples.append(VoiceSample(path, duration, rms, quality_score))
    self._prune()
    return path
