def is_protocol(__tp: type) -> bool:
    """Return True if the given type is a Protocol.

    Example::

        >>> from typing_extensions import Protocol, is_protocol
        >>> class P(Protocol):
        ...     def a(self) -> str: ...
        ...     b: int
        >>> is_protocol(P)
        True
        >>> is_protocol(int)
        False
    """
    return (
        isinstance(__tp, type)
        and getattr(__tp, '_is_protocol', False)
        and __tp is not Protocol
        and __tp is not getattr(typing, "Protocol", object())
    )

def get_protocol_members(__tp: type) -> typing.FrozenSet[str]:
    """Return the set of members defined in a Protocol.

    Example::

        >>> from typing_extensions import Protocol, get_protocol_members
        >>> class P(Protocol):
        ...     def a(self) -> str: ...
        ...     b: int
        >>> get_protocol_members(P)
        frozenset({'a', 'b'})

    Raise a TypeError for arguments that are not Protocols.
    """
    if not is_protocol(__tp):
        raise TypeError(f'{__tp!r} is not a Protocol')
    if hasattr(__tp, '__protocol_attrs__'):
        return frozenset(__tp.__protocol_attrs__)
    return frozenset(_get_protocol_attrs(__tp))


