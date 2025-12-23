def _convolve(self, img: np.ndarray, kernel: np.ndarray) -> np.ndarray:
from scipy.signal import convolve2d
