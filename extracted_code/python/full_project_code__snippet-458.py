    import threading

    from pip._vendor.tenacity import RetryCallState


class stop_base(abc.ABC):
    """Abstract base class for stop strategies."""

    @abc.abstractmethod
    def __call__(self, retry_state: "RetryCallState") -> bool:
        pass

    def __and__(self, other: "stop_base") -> "stop_all":
        return stop_all(self, other)

    def __or__(self, other: "stop_base") -> "stop_any":
        return stop_any(self, other)


