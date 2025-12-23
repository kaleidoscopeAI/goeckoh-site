"""Return an adapter factory for `ob` from `registry`"""
types = _always_object(inspect.getmro(getattr(ob, '__class__', type(ob))))
for t in types:
    if t in registry:
        return registry[t]


