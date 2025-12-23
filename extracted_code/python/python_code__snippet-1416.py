class _AnyMeta(type):
    def __instancecheck__(self, obj):
        if self is Any:
            raise TypeError("typing_extensions.Any cannot be used with isinstance()")
        return super().__instancecheck__(obj)

    def __repr__(self):
        if self is Any:
            return "typing_extensions.Any"
        return super().__repr__()

class Any(metaclass=_AnyMeta):
    """Special type indicating an unconstrained type.
    - Any is compatible with every type.
    - Any assumed to have all methods.
    - All values assumed to be instances of Any.
    Note that all the above statements are true from the point of view of
    static type checkers. At runtime, Any should not be used with instance
    checks.
    """
    def __new__(cls, *args, **kwargs):
        if cls is Any:
            raise TypeError("Any cannot be instantiated")
        return super().__new__(cls, *args, **kwargs)


