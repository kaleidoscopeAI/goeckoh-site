# Add default parameter - PEP 696
class TypeVarTuple(metaclass=_TypeVarLikeMeta):
    """Type variable tuple."""

    _backported_typevarlike = typing.TypeVarTuple

    def __new__(cls, name, *, default=_marker):
        tvt = typing.TypeVarTuple(name)
        _set_default(tvt, default)
        _set_module(tvt)
        return tvt

    def __init_subclass__(self, *args, **kwds):
        raise TypeError("Cannot subclass special typing classes")

