@runtime_checkable
class SupportsInt(Protocol):
    """An ABC with one abstract method __int__."""
    __slots__ = ()

    @abc.abstractmethod
    def __int__(self) -> int:
        pass

@runtime_checkable
class SupportsFloat(Protocol):
    """An ABC with one abstract method __float__."""
    __slots__ = ()

    @abc.abstractmethod
    def __float__(self) -> float:
        pass

@runtime_checkable
class SupportsComplex(Protocol):
    """An ABC with one abstract method __complex__."""
    __slots__ = ()

    @abc.abstractmethod
    def __complex__(self) -> complex:
        pass

@runtime_checkable
class SupportsBytes(Protocol):
    """An ABC with one abstract method __bytes__."""
    __slots__ = ()

    @abc.abstractmethod
    def __bytes__(self) -> bytes:
        pass

@runtime_checkable
class SupportsIndex(Protocol):
    __slots__ = ()

    @abc.abstractmethod
    def __index__(self) -> int:
        pass

@runtime_checkable
class SupportsAbs(Protocol[T_co]):
    """
    An ABC with one abstract method __abs__ that is covariant in its return type.
    """
    __slots__ = ()

    @abc.abstractmethod
    def __abs__(self) -> T_co:
        pass

@runtime_checkable
class SupportsRound(Protocol[T_co]):
    """
    An ABC with one abstract method __round__ that is covariant in its return type.
    """
    __slots__ = ()

    @abc.abstractmethod
    def __round__(self, ndigits: int = 0) -> T_co:
        pass


