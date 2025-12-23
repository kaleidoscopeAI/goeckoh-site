"""Combine several waiting strategies."""

def __init__(self, *strategies: wait_base) -> None:
    self.wait_funcs = strategies

def __call__(self, retry_state: "RetryCallState") -> float:
    return sum(x(retry_state=retry_state) for x in self.wait_funcs)


