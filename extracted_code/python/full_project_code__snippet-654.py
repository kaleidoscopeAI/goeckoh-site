"""
Acts like a functools.partial, but can be edited. In other words, it represents a type that hasn't yet been
constructed.
"""

# We need this here because the railroad constructors actually transform the data, so can't be called until the
# entire tree is assembled

def __init__(self, func: Callable[..., T], args: list, kwargs: dict):
    self.func = func
    self.args = args
    self.kwargs = kwargs

@classmethod
def from_call(cls, func: Callable[..., T], *args, **kwargs) -> "EditablePartial[T]":
    """
    If you call this function in the same way that you would call the constructor, it will store the arguments
    as you expect. For example EditablePartial.from_call(Fraction, 1, 3)() == Fraction(1, 3)
    """
    return EditablePartial(func=func, args=list(args), kwargs=kwargs)

@property
def name(self):
    return self.kwargs["name"]

def __call__(self) -> T:
    """
    Evaluate the partial and return the result
    """
    args = self.args.copy()
    kwargs = self.kwargs.copy()

    # This is a helpful hack to allow you to specify varargs parameters (e.g. *args) as keyword args (e.g.
    # args=['list', 'of', 'things'])
    arg_spec = inspect.getfullargspec(self.func)
    if arg_spec.varargs in self.kwargs:
        args += kwargs.pop(arg_spec.varargs)

    return self.func(*args, **kwargs)


