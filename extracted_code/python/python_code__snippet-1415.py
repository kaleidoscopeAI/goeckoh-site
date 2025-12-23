def _should_collect_from_parameters(t):
    return isinstance(t, typing._GenericAlias) and not t._special


