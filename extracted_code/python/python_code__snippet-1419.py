# @final exists in 3.8+, but we backport it for all versions
# before 3.11 to keep support for the __final__ attribute.
# See https://bugs.python.org/issue46342
def final(f):
    """This decorator can be used to indicate to type checkers that
    the decorated method cannot be overridden, and decorated class
    cannot be subclassed. For example:

        class Base:
            @final
            def done(self) -> None:
                ...
        class Sub(Base):
            def done(self) -> None:  # Error reported by type checker
                ...
        @final
        class Leaf:
            ...
        class Other(Leaf):  # Error reported by type checker
            ...

    There is no runtime checking of these properties. The decorator
    sets the ``__final__`` attribute to ``True`` on the decorated object
    to allow runtime introspection.
    """
    try:
        f.__final__ = True
    except (AttributeError, TypeError):
        # Skip the attribute silently if it is not writable.
        # AttributeError happens if the object has __slots__ or a
        # read-only property, TypeError if it's a builtin class.
        pass
    return f


