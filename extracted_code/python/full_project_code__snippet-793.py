"""Stop when the time from the first attempt >= limit."""

def __init__(self, max_delay: _utils.time_unit_type) -> None:
    self.max_delay = _utils.to_seconds(max_delay)

def __call__(self, retry_state: "RetryCallState") -> bool:
    if retry_state.seconds_since_start is None:
        raise RuntimeError("__call__() called but seconds_since_start is not set")
    return retry_state.seconds_since_start >= self.max_delay


