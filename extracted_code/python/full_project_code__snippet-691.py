# In a future version, uncomment the code in the internal _inner() functions
# to begin emitting DeprecationWarnings.

# Unwrap staticmethod/classmethod
fn = getattr(fn, "__func__", fn)

# (Presence of 'self' arg in signature is used by explain_exception() methods, so we take
# some extra steps to add it if present in decorated function.)
if "self" == list(inspect.signature(fn).parameters)[0]:

    @wraps(fn)
    def _inner(self, *args, **kwargs):
        # warnings.warn(
        #     f"Deprecated - use {fn.__name__}", DeprecationWarning, stacklevel=3
        # )
        return fn(self, *args, **kwargs)

else:

    @wraps(fn)
    def _inner(*args, **kwargs):
        # warnings.warn(
        #     f"Deprecated - use {fn.__name__}", DeprecationWarning, stacklevel=3
        # )
        return fn(*args, **kwargs)

_inner.__doc__ = f"""Deprecated - use :class:`{fn.__name__}`"""
_inner.__name__ = compat_name
_inner.__annotations__ = fn.__annotations__
if isinstance(fn, types.FunctionType):
    _inner.__kwdefaults__ = fn.__kwdefaults__
elif isinstance(fn, type) and hasattr(fn, "__init__"):
    _inner.__kwdefaults__ = fn.__init__.__kwdefaults__
else:
    _inner.__kwdefaults__ = None
_inner.__qualname__ = fn.__qualname__
return cast(C, _inner)


