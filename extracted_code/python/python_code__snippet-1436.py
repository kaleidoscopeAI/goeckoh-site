# Inherits from list as a workaround for Callable checks in Python < 3.9.2.
class ParamSpec(list, _DefaultMixin):
    """Parameter specification variable.

    Usage::

       P = ParamSpec('P')

    Parameter specification variables exist primarily for the benefit of static
    type checkers.  They are used to forward the parameter types of one
    callable to another callable, a pattern commonly found in higher order
    functions and decorators.  They are only valid when used in ``Concatenate``,
    or s the first argument to ``Callable``. In Python 3.10 and higher,
    they are also supported in user-defined Generics at runtime.
    See class Generic for more information on generic types.  An
    example for annotating a decorator::

       T = TypeVar('T')
       P = ParamSpec('P')

       def add_logging(f: Callable[P, T]) -> Callable[P, T]:
           '''A type-safe decorator to add logging to a function.'''
           def inner(*args: P.args, **kwargs: P.kwargs) -> T:
               logging.info(f'{f.__name__} was called')
               return f(*args, **kwargs)
           return inner

       @add_logging
       def add_two(x: float, y: float) -> float:
           '''Add two numbers together.'''
           return x + y

    Parameter specification variables defined with covariant=True or
    contravariant=True can be used to declare covariant or contravariant
    generic types.  These keyword arguments are valid, but their actual semantics
    are yet to be decided.  See PEP 612 for details.

    Parameter specification variables can be introspected. e.g.:

       P.__name__ == 'T'
       P.__bound__ == None
       P.__covariant__ == False
       P.__contravariant__ == False

    Note that only parameter specification variables defined in global scope can
    be pickled.
    """

    # Trick Generic __parameters__.
    __class__ = typing.TypeVar

    @property
    def args(self):
        return ParamSpecArgs(self)

    @property
    def kwargs(self):
        return ParamSpecKwargs(self)

    def __init__(self, name, *, bound=None, covariant=False, contravariant=False,
                 infer_variance=False, default=_marker):
        super().__init__([self])
        self.__name__ = name
        self.__covariant__ = bool(covariant)
        self.__contravariant__ = bool(contravariant)
        self.__infer_variance__ = bool(infer_variance)
        if bound:
            self.__bound__ = typing._type_check(bound, 'Bound must be a type.')
        else:
            self.__bound__ = None
        _DefaultMixin.__init__(self, default)

        # for pickling:
        def_mod = _caller()
        if def_mod != 'typing_extensions':
            self.__module__ = def_mod

    def __repr__(self):
        if self.__infer_variance__:
            prefix = ''
        elif self.__covariant__:
            prefix = '+'
        elif self.__contravariant__:
            prefix = '-'
        else:
            prefix = '~'
        return prefix + self.__name__

    def __hash__(self):
        return object.__hash__(self)

    def __eq__(self, other):
        return self is other

    def __reduce__(self):
        return self.__name__

    # Hack to get typing._type_check to pass.
    def __call__(self, *args, **kwargs):
        pass


