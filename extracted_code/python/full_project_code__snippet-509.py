def __init__(self, name):
    super(_LazyModule, self).__init__(name)
    self.__doc__ = self.__class__.__doc__

def __dir__(self):
    attrs = ["__doc__", "__name__"]
    attrs += [attr.name for attr in self._moved_attributes]
    return attrs

# Subclasses should override this
_moved_attributes = []


