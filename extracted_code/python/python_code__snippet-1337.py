"""
A class representing an in-package resource, such as a data file. This is
not normally instantiated by user code, but rather by a
:class:`ResourceFinder` which manages the resource.
"""
is_container = False        # Backwards compatibility

def as_stream(self):
    """
    Get the resource as a stream.

    This is not a property to make it obvious that it returns a new stream
    each time.
    """
    return self.finder.get_stream(self)

@cached_property
def file_path(self):
    global cache
    if cache is None:
        cache = ResourceCache()
    return cache.get(self)

@cached_property
def bytes(self):
    return self.finder.get_bytes(self)

@cached_property
def size(self):
    return self.finder.get_size(self)


