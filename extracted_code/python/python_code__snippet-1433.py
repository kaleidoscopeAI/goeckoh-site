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


