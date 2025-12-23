"""Abstract base class for wait strategies."""

@abc.abstractmethod
def __call__(self, retry_state: "RetryCallState") -> float:
    pass

def __add__(self, other: "wait_base") -> "wait_combine":
    return wait_combine(self, other)

def __radd__(self, other: "wait_base") -> typing.Union["wait_combine", "wait_base"]:
    # make it possible to use multiple waits with the built-in sum function
    if other == 0:  # type: ignore[comparison-overlap]
        return self
    return self.__add__(other)


