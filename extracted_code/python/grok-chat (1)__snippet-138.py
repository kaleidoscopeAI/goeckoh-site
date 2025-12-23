base_dir: Path
samples: Dict[Style, List[VoiceSample]] = field(default_factory=lambda: {"neutral": [], "calm": [], "excited": []})
max_samples_per_style: int = 32

def _compute_rms(self, wav: np.ndarray) -> float:
    if wav.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(wav))))

def add_sample_from_wav(self, wav: np.ndarray, style: Style, name: Optional[str] = None) -> Path:
    self.base_dir.mkdir(parents=True, exist_ok=True)
    if name is None:
        name = time.strftime("%Y%m%d_%H%M%S")
    out_path = self.base_dir / f"{style}_{name}.wav"
    sf.write(out_path, wav, 16000)
    rms = self._compute_rms(wav)
    duration_s = float(len(wav) / 16000.0)
    sample = VoiceSample(
        path=out_path,
        style=style,
        rms=rms,
        duration_s=duration_s,
        added_ts=time.time(),
    )
    self.samples[style].append(sample)
    # Slow drift: prune oldest if too many
    if len(self.samples[style]) > self.max_samples_per_style:
        self.samples[style].sort(key=lambda s: s.added_ts)
        self.samples[style] = self.samples[style][-self.max_samples_per_style:]
    return out_path

def load_existing(self) -> None:
    if not self.base_dir.exists():
        return
    for wav_path in self.base_dir.glob("*.wav"):
        stem = wav_path.stem
        if "_" not in stem:
            continue
        style_str, _ = stem.split("_", 1)
        if style_str not in ("neutral", "calm", "excited"):
            continue
        style: Style = style_str  # type: ignore
        wav, sr = sf.read(wav_path)
        rms = self._compute_rms(wav)
        duration_s = float(len(wav) / sr)
        sample = VoiceSample(
            path=wav_path,
            style=style,
            rms=rms,
            duration_s=duration_s,
            added_ts=wav_path.stat().st_mtime,
        )
        self.samples[style].append(sample)

def pick_reference(self, style: Style, min_duration_s: float = 1.0) -> Optional[Path]:
    candidates = [s for s in self.samples.get(style, []) if s.duration_s >= min_duration_s]
    if not candidates:
        return None
    # Pick random for variety
    return random.choice(candidates).path

def maybe_adapt_from_attempt(self, attempt_wav: np.ndarray, style: Style, quality_score: float = 1.0) -> None:
    if quality_score >= 0.9:  # only add very good attempts
        self.add_sample_from_wav(attempt_wav, style, name=None)

