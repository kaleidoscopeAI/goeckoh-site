"""
Resolves the argument to a numeric constant, which can be passed to
the wrap_socket function/method from the ssl module.
Defaults to :data:`ssl.CERT_REQUIRED`.
If given a string it is assumed to be the name of the constant in the
:mod:`ssl` module or its abbreviation.
(So you can specify `REQUIRED` instead of `CERT_REQUIRED`.
If it's neither `None` nor a string we assume it is already the numeric
constant which can directly be passed to wrap_socket.
"""
if candidate is None:
    return CERT_REQUIRED

if isinstance(candidate, str):
    res = getattr(ssl, candidate, None)
    if res is None:
        res = getattr(ssl, "CERT_" + candidate)
    return res

return candidate


