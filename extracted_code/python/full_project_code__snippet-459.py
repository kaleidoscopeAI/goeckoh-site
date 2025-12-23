class stop_any(stop_base):
    """Stop if any of the stop condition is valid."""

    def __init__(self, *stops: stop_base) -> None:
        self.stops = stops

    def __call__(self, retry_state: "RetryCallState") -> bool:
        return any(x(retry_state) for x in self.stops)


class stop_all(stop_base):
    """Stop if all the stop conditions are valid."""

    def __init__(self, *stops: stop_base) -> None:
        self.stops = stops

    def __call__(self, retry_state: "RetryCallState") -> bool:
        return all(x(retry_state) for x in self.stops)


class _stop_never(stop_base):
    """Never stop."""

    def __call__(self, retry_state: "RetryCallState") -> bool:
        return False


