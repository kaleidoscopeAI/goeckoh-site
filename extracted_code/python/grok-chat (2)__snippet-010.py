import librosa
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean

class SimilarityScorer:
    def __init__(self, sample_rate: int = 16000) -> None:
        self.sample_rate = sample_rate

    def compare(self, ref_path: Path, test_path: Path) -> float:
        ref, _ = librosa.load(ref_path, sr=self.sample_rate)
        test, _ = librosa.load(test_path, sr=self.sample_rate)
        mfcc_ref = librosa.feature.mfcc(y=ref, sr=self.sample_rate)
        mfcc_test = librosa.feature.mfcc(y=test, sr=self.sample_rate)
        distance, _ = fastdtw(mfcc_ref.T, mfcc_test.T, dist=euclidean)
        similarity = 1 / (1 + distance)
        return similarity
