# from jaraco.functools 1.3
def _call_aside(f, *args, **kwargs):
    f(*args, **kwargs)
    return f


