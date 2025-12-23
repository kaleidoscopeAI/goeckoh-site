def runtime_checkable(cls):
    """Mark a protocol class as a runtime protocol, so that it
    can be used with isinstance() and issubclass(). Raise TypeError
    if applied to a non-protocol class.

    This allows a simple-minded structural check very similar to the
    one-offs in collections.abc such as Hashable.
    """
    if not (
        (isinstance(cls, _ProtocolMeta) or issubclass(cls, typing.Generic))
        and getattr(cls, "_is_protocol", False)
    ):
        raise TypeError('@runtime_checkable can be only applied to protocol classes,'
                        f' got {cls!r}')
    cls._is_runtime_protocol = True
    return cls


