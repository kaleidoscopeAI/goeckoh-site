645
646 +    def _load_reference_audio(self, sr: int) -> np.ndarray:
647 +        """Load the current reference wav into float32 at target sr."""
648 +        if not (self._reference_wav and self._reference_wav.exists()):
649 +            return np.zeros(1, dtype=np.float32)
650 +        try:
651 +            import soundfile as sf  # optional dependency
652 +        except Exception:
653 +            return np.zeros(1, dtype=np.float32)
654 +        try:
655 +            data, file_sr = sf.read(str(self._reference_wav), dtype="fl
     oat32")
656 +            if data.ndim > 1:
657 +                data = data[:, 0]
658 +            if file_sr != sr:
659 +                data = self._resample(np.asarray(data, dtype=np.float32
     ), file_sr, sr)
660 +            return np.asarray(data, dtype=np.float32)
661 +        except Exception:
662 +            return np.zeros(1, dtype=np.float32)
663 +
664      def _setup_xtts(self) -> None:

