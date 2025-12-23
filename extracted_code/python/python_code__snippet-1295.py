"""Retries if any of the retries condition is valid."""

def __init__(self, *retries: retry_base) -> None:
    self.retries = retries

def __call__(self, retry_state: "RetryCallState") -> bool:
    return any(r(retry_state) for r in self.retries)


