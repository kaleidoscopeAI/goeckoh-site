"""Stop when the given event is set."""

def __init__(self, event: "threading.Event") -> None:
    self.event = event

def __call__(self, retry_state: "RetryCallState") -> bool:
    return self.event.is_set()


