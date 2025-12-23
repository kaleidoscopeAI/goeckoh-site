def __init__(self, func):
    self.func = func
    # for attr in ('__name__', '__module__', '__doc__'):
    #     setattr(self, attr, getattr(func, attr, None))

def __get__(self, obj, cls=None):
    if obj is None:
        return self
    value = self.func(obj)
    object.__setattr__(obj, self.func.__name__, value)
    # obj.__dict__[self.func.__name__] = value = self.func(obj)
    return value


