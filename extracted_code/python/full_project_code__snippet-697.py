'''
Wraps a stream (such as stdout), acting as a transparent proxy for all
attribute access apart from method 'write()', which is delegated to our
Converter instance.
'''
def __init__(self, wrapped, converter):
    # double-underscore everything to prevent clashes with names of
    # attributes on the wrapped stream object.
    self.__wrapped = wrapped
    self.__convertor = converter

def __getattr__(self, name):
    return getattr(self.__wrapped, name)

def __enter__(self, *args, **kwargs):
    # special method lookup bypasses __getattr__/__getattribute__, see
    # https://stackoverflow.com/questions/12632894/why-doesnt-getattr-work-with-exit
    # thus, contextlib magic methods are not proxied via __getattr__
    return self.__wrapped.__enter__(*args, **kwargs)

def __exit__(self, *args, **kwargs):
    return self.__wrapped.__exit__(*args, **kwargs)

def __setstate__(self, state):
    self.__dict__ = state

def __getstate__(self):
    return self.__dict__

def write(self, text):
    self.__convertor.write(text)

def isatty(self):
    stream = self.__wrapped
    if 'PYCHARM_HOSTED' in os.environ:
        if stream is not None and (stream is sys.__stdout__ or stream is sys.__stderr__):
            return True
    try:
        stream_isatty = stream.isatty
    except AttributeError:
        return False
    else:
        return stream_isatty()

@property
def closed(self):
    stream = self.__wrapped
    try:
        return stream.closed
    # AttributeError in the case that the stream doesn't support being closed
    # ValueError for the case that the stream has already been detached when atexit runs
    except (AttributeError, ValueError):
        return True


