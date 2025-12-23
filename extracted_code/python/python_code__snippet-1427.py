def assert_type(__val, __typ):
    """Assert (to the type checker) that the value is of the given type.

    When the type checker encounters a call to assert_type(), it
    emits an error if the value is not of the specified type::

        def greet(name: str) -> None:
            assert_type(name, str)  # ok
            assert_type(name, int)  # type checker error

    At runtime this returns the first argument unchanged and otherwise
    does nothing.
    """
    return __val


