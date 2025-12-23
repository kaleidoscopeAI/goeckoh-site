Pythonfrom typing import Optional, List
from collections import deque

class BehaviorMonitor:
    def __init__(self, history_len: int = 5, correction_streak_thresh: int = 3, repeat_thresh: int = 3, high_rms_thresh: float = 0.1) -> None:
        self.history = deque(maxlen=history_len)
        self.correction_streak = 0
        self.correction_streak_thresh = correction_streak_thresh
        self.repeat_thresh = repeat_thresh
        self.high_rms_thresh = high_rms_thresh

    def register(self, normalized_text: str, needs_correction: bool, rms: float) -> Optional[str]:
        self.history.append(normalized_text)
        if needs_correction:
            self.correction_streak += 1
        else:
            self.correction_streak = 0

        if self.correction_streak >= self.correction_streak_thresh:
            return \"anxious\"

        repeats = list(self.history).count(normalized_text)
        if repeats >= self.repeat_thresh:
            return \"perseveration\"

        if rms >= self.high_rms_thresh:
            return \"high_energy\"

        if self.correction_streak == 0:
            return \"encouragement\"

        return None
