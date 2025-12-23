# This does exactly the same what the :func:`py3:functools.update_wrapper`
# function does on Python versions after 3.2. It sets the ``__wrapped__``
# attribute on ``wrapper`` object and it doesn't raise an error if any of
# the attributes mentioned in ``assigned`` and ``updated`` are missing on
# ``wrapped`` object.
def _update_wrapper(
    wrapper,
    wrapped,
    assigned=functools.WRAPPER_ASSIGNMENTS,
    updated=functools.WRAPPER_UPDATES,
):
    for attr in assigned:
        try:
            value = getattr(wrapped, attr)
        except AttributeError:
            continue
        else:
            setattr(wrapper, attr, value)
    for attr in updated:
        getattr(wrapper, attr).update(getattr(wrapped, attr, {}))
    wrapper.__wrapped__ = wrapped
    return wrapper

_update_wrapper.__doc__ = functools.update_wrapper.__doc__

def wraps(
    wrapped,
    assigned=functools.WRAPPER_ASSIGNMENTS,
    updated=functools.WRAPPER_UPDATES,
):
    return functools.partial(
        _update_wrapper, wrapped=wrapped, assigned=assigned, updated=updated
    )

wraps.__doc__ = functools.wraps.__doc__

