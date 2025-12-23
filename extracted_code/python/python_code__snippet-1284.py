"""Retry strategy that never rejects any result."""

def __call__(self, retry_state: "RetryCallState") -> bool:
    return False


