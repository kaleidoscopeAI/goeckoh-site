"""Sleep strategy that waits on an event to be set."""

def __init__(self, event: "threading.Event") -> None:
    self.event = event

def __call__(self, timeout: typing.Optional[float]) -> None:
    # NOTE(harlowja): this may *not* actually wait for timeout
    # seconds if the event is set (ie this may eject out early).
    self.event.wait(timeout=timeout)


