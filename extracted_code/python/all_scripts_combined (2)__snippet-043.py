class SimilarityScorer:
    settings: AudioSettings

    def _load_mfcc(self, path: Path) -> np.ndarray:
        audio, sr = librosa.load(path, sr=self.settings.sample_rate)
        return librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=20)

    def compare(self, reference: Path, attempt: Path) -> float:
        ref_mfcc = self._load_mfcc(reference)
        att_mfcc = self._load_mfcc(attempt)
        distance, _ = fastdtw(ref_mfcc.T, att_mfcc.T)
        return 1.0 / (1.0 + distance)
