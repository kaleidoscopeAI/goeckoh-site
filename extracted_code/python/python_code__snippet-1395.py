"""
A class implementing a cache for resources that need to live in the file system
e.g. shared libraries. This class was moved from resources to here because it
could be used by other modules, e.g. the wheel module.
"""

def __init__(self, base):
    """
    Initialise an instance.

    :param base: The base directory where the cache should be located.
    """
    # we use 'isdir' instead of 'exists', because we want to
    # fail if there's a file with that name
    if not os.path.isdir(base):  # pragma: no cover
        os.makedirs(base)
    if (os.stat(base).st_mode & 0o77) != 0:
        logger.warning('Directory \'%s\' is not private', base)
    self.base = os.path.abspath(os.path.normpath(base))

def prefix_to_dir(self, prefix):
    """
    Converts a resource prefix to a directory name in the cache.
    """
    return path_to_cache_dir(prefix)

def clear(self):
    """
    Clear the cache.
    """
    not_removed = []
    for fn in os.listdir(self.base):
        fn = os.path.join(self.base, fn)
        try:
            if os.path.islink(fn) or os.path.isfile(fn):
                os.remove(fn)
            elif os.path.isdir(fn):
                shutil.rmtree(fn)
        except Exception:
            not_removed.append(fn)
    return not_removed


