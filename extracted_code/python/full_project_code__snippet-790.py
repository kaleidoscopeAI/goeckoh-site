"""Never stop."""

def __call__(self, retry_state: "RetryCallState") -> bool:
    return False


