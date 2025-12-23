class stop_when_event_set(stop_base):
    """Stop when the given event is set."""

    def __init__(self, event: "threading.Event") -> None:
        self.event = event

    def __call__(self, retry_state: "RetryCallState") -> bool:
        return self.event.is_set()


class stop_after_attempt(stop_base):
    """Stop when the previous attempt >= max_attempt."""

    def __init__(self, max_attempt_number: int) -> None:
        self.max_attempt_number = max_attempt_number

    def __call__(self, retry_state: "RetryCallState") -> bool:
        return retry_state.attempt_number >= self.max_attempt_number


class stop_after_delay(stop_base):
    """Stop when the time from the first attempt >= limit."""

    def __init__(self, max_delay: _utils.time_unit_type) -> None:
        self.max_delay = _utils.to_seconds(max_delay)

    def __call__(self, retry_state: "RetryCallState") -> bool:
        if retry_state.seconds_since_start is None:
            raise RuntimeError("__call__() called but seconds_since_start is not set")
        return retry_state.seconds_since_start >= self.max_delay


