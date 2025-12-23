<parameter name="code">import numpy as np
import time

def is_actual_speech(audio_np: np.ndarray, threshold_db=-35, min_speech_ms=400) -> bool:
