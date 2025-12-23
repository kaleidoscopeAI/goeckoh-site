# Python < 3.8
class cached_property:  # type: ignore
    """A version of @property which caches the value.  On access, it calls the
    underlying function and sets the value in `__dict__` so future accesses
    will not re-call the property.
    """

    def __init__(self, f: Callable[[Any], Any]) -> None:
        self._fname = f.__name__
        self._f = f

    def __get__(self, obj: Any, owner: Type[Any]) -> Any:
        assert obj is not None, f"call {self._fname} on an instance"
        ret = obj.__dict__[self._fname] = self._f(obj)
        return ret


