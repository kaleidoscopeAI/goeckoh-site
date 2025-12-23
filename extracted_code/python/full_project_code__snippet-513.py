"""Remove item from six.moves."""
try:
    delattr(_MovedItems, name)
except AttributeError:
    try:
        del moves.__dict__[name]
    except KeyError:
        raise AttributeError("no such move, %r" % (name,))


