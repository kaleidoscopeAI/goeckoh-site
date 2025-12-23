"""Helper function used in Protocol.__init_subclass__ and _TypedDictMeta.__new__.

The contents of this function are very similar
to logic found in typing.Generic.__init_subclass__
on the CPython main branch.
"""
tvars = []
if '__orig_bases__' in cls.__dict__:
    tvars = _collect_type_vars(cls.__orig_bases__)
    # Look for Generic[T1, ..., Tn] or Protocol[T1, ..., Tn].
    # If found, tvars must be a subset of it.
    # If not found, tvars is it.
    # Also check for and reject plain Generic,
    # and reject multiple Generic[...] and/or Protocol[...].
    gvars = None
    for base in cls.__orig_bases__:
        if (isinstance(base, typing._GenericAlias) and
                base.__origin__ in (typing.Generic, Protocol)):
            # for error messages
            the_base = base.__origin__.__name__
            if gvars is not None:
                raise TypeError(
                    "Cannot inherit from Generic[...]"
                    " and/or Protocol[...] multiple types.")
            gvars = base.__parameters__
    if gvars is None:
        gvars = tvars
    else:
        tvarset = set(tvars)
        gvarset = set(gvars)
        if not tvarset <= gvarset:
            s_vars = ', '.join(str(t) for t in tvars if t not in gvarset)
            s_args = ', '.join(str(g) for g in gvars)
            raise TypeError(f"Some type variables ({s_vars}) are"
                            f" not listed in {the_base}[{s_args}]")
        tvars = gvars
cls.__parameters__ = tuple(tvars)


