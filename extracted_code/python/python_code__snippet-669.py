def _set_default(type_param, default):
    if isinstance(default, (tuple, list)):
        type_param.__default__ = tuple((typing._type_check(d, "Default must be a type")
                                        for d in default))
    elif default != _marker:
        type_param.__default__ = typing._type_check(default, "Default must be a type")
    else:
        type_param.__default__ = None


def _set_module(typevarlike):
    # for pickling:
    def_mod = _caller(depth=3)
    if def_mod != 'typing_extensions':
        typevarlike.__module__ = def_mod


class _DefaultMixin:
    """Mixin for TypeVarLike defaults."""

    __slots__ = ()
    __init__ = _set_default


# Classes using this metaclass must provide a _backported_typevarlike ClassVar
class _TypeVarLikeMeta(type):
    def __instancecheck__(cls, __instance: Any) -> bool:
        return isinstance(__instance, cls._backported_typevarlike)


# Add default and infer_variance parameters from PEP 696 and 695
class TypeVar(metaclass=_TypeVarLikeMeta):
    """Type variable."""

    _backported_typevarlike = typing.TypeVar

    def __new__(cls, name, *constraints, bound=None,
                covariant=False, contravariant=False,
                default=_marker, infer_variance=False):
        if hasattr(typing, "TypeAliasType"):
            # PEP 695 implemented, can pass infer_variance to typing.TypeVar
            typevar = typing.TypeVar(name, *constraints, bound=bound,
                                     covariant=covariant, contravariant=contravariant,
                                     infer_variance=infer_variance)
        else:
            typevar = typing.TypeVar(name, *constraints, bound=bound,
                                     covariant=covariant, contravariant=contravariant)
            if infer_variance and (covariant or contravariant):
                raise ValueError("Variance cannot be specified with infer_variance.")
            typevar.__infer_variance__ = infer_variance
        _set_default(typevar, default)
        _set_module(typevar)
        return typevar

    def __init_subclass__(cls) -> None:
        raise TypeError(f"type '{__name__}.TypeVar' is not an acceptable base type")


