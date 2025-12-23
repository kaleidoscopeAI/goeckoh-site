"""
A representation of the tag triple for a wheel.

Instances are considered immutable and thus are hashable. Equality checking
is also supported.
"""

__slots__ = ["_interpreter", "_abi", "_platform", "_hash"]

def __init__(self, interpreter: str, abi: str, platform: str) -> None:
    self._interpreter = interpreter.lower()
    self._abi = abi.lower()
    self._platform = platform.lower()
    # The __hash__ of every single element in a Set[Tag] will be evaluated each time
    # that a set calls its `.disjoint()` method, which may be called hundreds of
    # times when scanning a page of links for packages with tags matching that
    # Set[Tag]. Pre-computing the value here produces significant speedups for
    # downstream consumers.
    self._hash = hash((self._interpreter, self._abi, self._platform))

@property
def interpreter(self) -> str:
    return self._interpreter

@property
def abi(self) -> str:
    return self._abi

@property
def platform(self) -> str:
    return self._platform

def __eq__(self, other: object) -> bool:
    if not isinstance(other, Tag):
        return NotImplemented

    return (
        (self._hash == other._hash)  # Short-circuit ASAP for perf reasons.
        and (self._platform == other._platform)
        and (self._abi == other._abi)
        and (self._interpreter == other._interpreter)
    )

def __hash__(self) -> int:
    return self._hash

def __str__(self) -> str:
    return f"{self._interpreter}-{self._abi}-{self._platform}"

def __repr__(self) -> str:
    return f"<{self} @ {id(self)}>"


