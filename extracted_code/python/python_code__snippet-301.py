    def is_consecutive(c):
        c_int = ord(c)
        is_consecutive.prev, prev = c_int, is_consecutive.prev
        if c_int - prev > 1:
            is_consecutive.value = next(is_consecutive.counter)
        return is_consecutive.value

    is_consecutive.prev = 0  # type: ignore [attr-defined]
    is_consecutive.counter = itertools.count()  # type: ignore [attr-defined]
    is_consecutive.value = -1  # type: ignore [attr-defined]

    def escape_re_range_char(c):
        return "\\" + c if c in r"\^-][" else c

    def no_escape_re_range_char(c):
        return c

    if not re_escape:
        escape_re_range_char = no_escape_re_range_char

    ret = []
    s = "".join(sorted(set(s)))
    if len(s) > 3:
        for _, chars in itertools.groupby(s, key=is_consecutive):
            first = last = next(chars)
            last = collections.deque(
                itertools.chain(iter([last]), chars), maxlen=1
            ).pop()
            if first == last:
                ret.append(escape_re_range_char(first))
            else:
                sep = "" if ord(last) == ord(first) + 1 else "-"
                ret.append(
                    f"{escape_re_range_char(first)}{sep}{escape_re_range_char(last)}"
                )
    else:
        ret = [escape_re_range_char(c) for c in s]

    return "".join(ret)


def _flatten(ll: list) -> list:
    ret = []
    for i in ll:
        if isinstance(i, list):
            ret.extend(_flatten(i))
        else:
            ret.append(i)
    return ret


def _make_synonym_function(compat_name: str, fn: C) -> C:
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


def replaced_by_pep8(fn: C) -> Callable[[Callable], C]:
    """
    Decorator for pre-PEP8 compatibility synonyms, to link them to the new function.
    """
    return lambda other: _make_synonym_function(other.__name__, fn)


