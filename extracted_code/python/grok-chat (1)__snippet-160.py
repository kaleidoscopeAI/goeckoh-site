base_dir: Path
samples: Dict[Style, List[VoiceSample]] = field(default_factory=lambda: {"neutral": [], "calm": [], "excited": []})
max_samples_per_style: int = 32

# ... (previous load/add methods, using wave for RMS instead of librosa)

def _compute_rms(self, wav_path: Path) -> float:
    with wave.open(str(wav_path), 'rb') as wf:
        signal = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16)
        if signal.size == 0:
            return 0.0
        return float(np.sqrt(np.mean(signal.astype(np.float32)**2)))

