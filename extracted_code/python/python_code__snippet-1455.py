_F = typing.TypeVar("_F", bound=typing.Callable[..., typing.Any])

def override(__arg: _F) -> _F:
    """Indicate that a method is intended to override a method in a base class.

    Usage:

        class Base:
            def method(self) -> None: ...
                pass

        class Child(Base):
            @override
            def method(self) -> None:
                super().method()

    When this decorator is applied to a method, the type checker will
    validate that it overrides a method with the same name on a base class.
    This helps prevent bugs that may occur when a base class is changed
    without an equivalent change to a child class.

    There is no runtime checking of these properties. The decorator
    sets the ``__override__`` attribute to ``True`` on the decorated object
    to allow runtime introspection.

    See PEP 698 for details.

    """
    try:
        __arg.__override__ = True
    except (AttributeError, TypeError):
        # Skip the attribute silently if it is not writable.
        # AttributeError happens if the object has __slots__ or a
        # read-only property, TypeError if it's a builtin class.
        pass
    return __arg


