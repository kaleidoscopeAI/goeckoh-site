audio: AudioSettings
base_dir: Path
max_samples_per_style: int = 32
samples: Dict[Style, List[VoiceSample]] = field(default_factory=lambda: {
    "neutral": [],
    "calm": [],
    "excited": [],
})

def __post_init__(self) -> None:
    self.base_dir.mkdir(parents=True, exist_ok=True)
    self.load_existing()

# ------------------- helpers -------------------
def _style_dir(self, style: Style) -> Path:
    return self.base_dir / style

def _compute_rms(self, wav: np.ndarray) -> float:
    if wav.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(wav))))

def _register_sample(self, path: Path, style: Style, wav: np.ndarray, quality_score: float) -> None:
    duration = len(wav) / float(self.audio.sample_rate)
    rms = self._compute_rms(wav)
    sample = VoiceSample(
        path=path,
        duration_s=duration,
        rms=rms,
        style=style,
        quality_score=quality_score,
        added_ts=path.stat().st_mtime,
    )
    self.samples.setdefault(style, []).append(sample)
    self._prune(style)

def load_existing(self) -> None:
    for style in ("neutral", "calm", "excited"):
        dir_path = self._style_dir(style)
        if not dir_path.exists():
            continue
        for wav_path in sorted(dir_path.glob("*.wav")):
            try:
                data, sr = sf.read(wav_path, dtype="float32")
            except Exception:
                continue
            if data.ndim > 1:
                data = np.mean(data, axis=1)
            if sr != self.audio.sample_rate:
                data = librosa.resample(data, orig_sr=sr, target_sr=self.audio.sample_rate)
            duration = len(data) / float(self.audio.sample_rate)
            rms = self._compute_rms(np.asarray(data, dtype=np.float32))
            sample = VoiceSample(
                path=wav_path,
                duration_s=duration,
                rms=rms,
                style=style,  # type: ignore[arg-type]
                quality_score=1.0,
                added_ts=wav_path.stat().st_mtime,
            )
            self.samples.setdefault(style, []).append(sample)

def _prune(self, style: Style) -> None:
    if len(self.samples[style]) <= self.max_samples_per_style:
        return
    self.samples[style].sort(key=lambda s: (s.quality_score, s.added_ts), reverse=True)
    self.samples[style] = self.samples[style][: self.max_samples_per_style]

def add_sample_from_wav(
    self,
    wav: np.ndarray,
    style: Style,
    name: Optional[str] = None,
    quality_score: float = 1.0,
) -> Path:
    if wav.ndim > 1:
        wav = np.mean(wav, axis=1)
    wav = np.asarray(wav, dtype=np.float32)
    style_dir = self._style_dir(style)
    style_dir.mkdir(parents=True, exist_ok=True)
    suffix = name or uuid.uuid4().hex[:8]
    path = style_dir / f"{style}_{suffix}.wav"
    sf.write(path, wav, self.audio.sample_rate)
    self._register_sample(path, style, wav, quality_score)
    return path

def pick_reference(self, style: Style = "neutral") -> Optional[VoiceSample]:
    candidates = self.samples.get(style) or []
    if candidates:
        return max(candidates, key=lambda s: s.quality_score)
    if style != "neutral" and self.samples["neutral"]:
        return max(self.samples["neutral"], key=lambda s: s.quality_score)
    for lst in self.samples.values():
        if lst:
            return max(lst, key=lambda s: s.quality_score)
    return None

def maybe_adapt_from_attempt(
    self,
    attempt_wav: np.ndarray,
    style: Style,
    quality_score: float,
    min_quality_bootstrap: float = 0.8,
    min_quality_refine: float = 0.9,
) -> Optional[Path]:
    has_profile = any(self.samples.values())
    threshold = min_quality_bootstrap if not has_profile else min_quality_refine
    if quality_score < threshold:
        return None
    return self.add_sample_from_wav(attempt_wav, style, quality_score=quality_score)


