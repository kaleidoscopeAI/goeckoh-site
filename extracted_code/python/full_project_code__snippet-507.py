def __init__(self, name):
    self.name = name

def __get__(self, obj, tp):
    result = self._resolve()
    setattr(obj, self.name, result)  # Invokes __set__.
    try:
        # This is a bit ugly, but it avoids running this again by
        # removing this descriptor.
        delattr(obj.__class__, self.name)
    except AttributeError:
        pass
    return result


