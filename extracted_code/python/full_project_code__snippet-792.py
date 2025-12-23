"""Stop when the previous attempt >= max_attempt."""

def __init__(self, max_attempt_number: int) -> None:
    self.max_attempt_number = max_attempt_number

def __call__(self, retry_state: "RetryCallState") -> bool:
    return retry_state.attempt_number >= self.max_attempt_number


