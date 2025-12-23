def __init__(self, size):
    self.not_in_cache = not_in_cache = object()
    cache = {}
    keyring = [object()] * size
    cache_get = cache.get
    cache_pop = cache.pop
    keyiter = itertools.cycle(range(size))

    def get(_, key):
        return cache_get(key, not_in_cache)

    def set_(_, key, value):
        cache[key] = value
        i = next(keyiter)
        cache_pop(keyring[i], None)
        keyring[i] = key

    def clear(_):
        cache.clear()
        keyring[:] = [object()] * size

    self.size = size
    self.get = types.MethodType(get, self)
    self.set = types.MethodType(set_, self)
    self.clear = types.MethodType(clear, self)


