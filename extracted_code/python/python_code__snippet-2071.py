def __init__(self, min_update_interval_seconds: float) -> None:
    self._min_update_interval_seconds = min_update_interval_seconds
    self._last_update: float = 0

def ready(self) -> bool:
    now = time.time()
    delta = now - self._last_update
    return delta >= self._min_update_interval_seconds

def reset(self) -> None:
    self._last_update = time.time()


