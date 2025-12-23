"""An abstract class - provides cache directories for data from links

:param cache_dir: The root of the cache.
"""

def __init__(self, cache_dir: str) -> None:
    super().__init__()
    assert not cache_dir or os.path.isabs(cache_dir)
    self.cache_dir = cache_dir or None

def _get_cache_path_parts(self, link: Link) -> List[str]:
    """Get parts of part that must be os.path.joined with cache_dir"""

    # We want to generate an url to use as our cache key, we don't want to
    # just re-use the URL because it might have other items in the fragment
    # and we don't care about those.
    key_parts = {"url": link.url_without_fragment}
    if link.hash_name is not None and link.hash is not None:
        key_parts[link.hash_name] = link.hash
    if link.subdirectory_fragment:
        key_parts["subdirectory"] = link.subdirectory_fragment

    # Include interpreter name, major and minor version in cache key
    # to cope with ill-behaved sdists that build a different wheel
    # depending on the python version their setup.py is being run on,
    # and don't encode the difference in compatibility tags.
    # https://github.com/pypa/pip/issues/7296
    key_parts["interpreter_name"] = interpreter_name()
    key_parts["interpreter_version"] = interpreter_version()

    # Encode our key url with sha224, we'll use this because it has similar
    # security properties to sha256, but with a shorter total output (and
    # thus less secure). However the differences don't make a lot of
    # difference for our use case here.
    hashed = _hash_dict(key_parts)

    # We want to nest the directories some to prevent having a ton of top
    # level directories where we might run out of sub directories on some
    # FS.
    parts = [hashed[:2], hashed[2:4], hashed[4:6], hashed[6:]]

    return parts

def _get_candidates(self, link: Link, canonical_package_name: str) -> List[Any]:
    can_not_cache = not self.cache_dir or not canonical_package_name or not link
    if can_not_cache:
        return []

    path = self.get_path_for_link(link)
    if os.path.isdir(path):
        return [(candidate, path) for candidate in os.listdir(path)]
    return []

def get_path_for_link(self, link: Link) -> str:
    """Return a directory to store cached items in for link."""
    raise NotImplementedError()

def get(
    self,
    link: Link,
    package_name: Optional[str],
    supported_tags: List[Tag],
) -> Link:
    """Returns a link to a cached item if it exists, otherwise returns the
    passed link.
    """
    raise NotImplementedError()


