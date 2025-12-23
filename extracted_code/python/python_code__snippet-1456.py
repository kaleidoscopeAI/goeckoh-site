_T = typing.TypeVar("_T")

def deprecated(
    __msg: str,
    *,
    category: typing.Optional[typing.Type[Warning]] = DeprecationWarning,
    stacklevel: int = 1,
) -> typing.Callable[[_T], _T]:
    """Indicate that a class, function or overload is deprecated.

    Usage:

        @deprecated("Use B instead")
        class A:
            pass

        @deprecated("Use g instead")
        def f():
            pass

        @overload
        @deprecated("int support is deprecated")
        def g(x: int) -> int: ...
        @overload
        def g(x: str) -> int: ...

    When this decorator is applied to an object, the type checker
    will generate a diagnostic on usage of the deprecated object.

    The warning specified by ``category`` will be emitted on use
    of deprecated objects. For functions, that happens on calls;
    for classes, on instantiation. If the ``category`` is ``None``,
    no warning is emitted. The ``stacklevel`` determines where the
    warning is emitted. If it is ``1`` (the default), the warning
    is emitted at the direct caller of the deprecated object; if it
    is higher, it is emitted further up the stack.

    The decorator sets the ``__deprecated__``
    attribute on the decorated object to the deprecation message
    passed to the decorator. If applied to an overload, the decorator
    must be after the ``@overload`` decorator for the attribute to
    exist on the overload as returned by ``get_overloads()``.

    See PEP 702 for details.

    """
    def decorator(__arg: _T) -> _T:
        if category is None:
            __arg.__deprecated__ = __msg
            return __arg
        elif isinstance(__arg, type):
            original_new = __arg.__new__
            has_init = __arg.__init__ is not object.__init__

            @functools.wraps(original_new)
            def __new__(cls, *args, **kwargs):
                warnings.warn(__msg, category=category, stacklevel=stacklevel + 1)
                if original_new is not object.__new__:
                    return original_new(cls, *args, **kwargs)
                # Mirrors a similar check in object.__new__.
                elif not has_init and (args or kwargs):
                    raise TypeError(f"{cls.__name__}() takes no arguments")
                else:
                    return original_new(cls)

            __arg.__new__ = staticmethod(__new__)
            __arg.__deprecated__ = __new__.__deprecated__ = __msg
            return __arg
        elif callable(__arg):
            @functools.wraps(__arg)
            def wrapper(*args, **kwargs):
                warnings.warn(__msg, category=category, stacklevel=stacklevel + 1)
                return __arg(*args, **kwargs)

            __arg.__deprecated__ = wrapper.__deprecated__ = __msg
            return wrapper
        else:
            raise TypeError(
                "@deprecated decorator with non-None category must be applied to "
                f"a class or callable, not {__arg!r}"
            )

    return decorator


