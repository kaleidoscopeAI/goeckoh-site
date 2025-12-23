try:
    # 3.9+
    from typing import _BaseGenericAlias
except ImportError:
    _BaseGenericAlias = typing._GenericAlias
try:
    # 3.9+
    from typing import GenericAlias as _typing_GenericAlias
except ImportError:
    _typing_GenericAlias = typing._GenericAlias

def get_origin(tp):
    """Get the unsubscripted version of a type.

    This supports generic types, Callable, Tuple, Union, Literal, Final, ClassVar
    and Annotated. Return None for unsupported types. Examples::

        get_origin(Literal[42]) is Literal
        get_origin(int) is None
        get_origin(ClassVar[int]) is ClassVar
        get_origin(Generic) is Generic
        get_origin(Generic[T]) is Generic
        get_origin(Union[T, int]) is Union
        get_origin(List[Tuple[T, T]][int]) == list
        get_origin(P.args) is P
    """
    if isinstance(tp, _AnnotatedAlias):
        return Annotated
    if isinstance(tp, (typing._GenericAlias, _typing_GenericAlias, _BaseGenericAlias,
                       ParamSpecArgs, ParamSpecKwargs)):
        return tp.__origin__
    if tp is typing.Generic:
        return typing.Generic
    return None

def get_args(tp):
    """Get type arguments with all substitutions performed.

    For unions, basic simplifications used by Union constructor are performed.
    Examples::
        get_args(Dict[str, int]) == (str, int)
        get_args(int) == ()
        get_args(Union[int, Union[T, int], str][int]) == (int, str)
        get_args(Union[int, Tuple[T, int]][str]) == (int, Tuple[str, int])
        get_args(Callable[[], T][int]) == ([], int)
    """
    if isinstance(tp, _AnnotatedAlias):
        return (tp.__origin__,) + tp.__metadata__
    if isinstance(tp, (typing._GenericAlias, _typing_GenericAlias)):
        if getattr(tp, "_special", False):
            return ()
        res = tp.__args__
        if get_origin(tp) is collections.abc.Callable and res[0] is not Ellipsis:
            res = (list(res[:-1]), res[-1])
        return res
    return ()


