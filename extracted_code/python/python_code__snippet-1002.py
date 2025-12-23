class MetadataFile:
    """Information about a core metadata file associated with a distribution."""

    hashes: Optional[Dict[str, str]]

    def __post_init__(self) -> None:
        if self.hashes is not None:
            assert all(name in _SUPPORTED_HASHES for name in self.hashes)


def supported_hashes(hashes: Optional[Dict[str, str]]) -> Optional[Dict[str, str]]:
    # Remove any unsupported hash types from the mapping. If this leaves no
    # supported hashes, return None
    if hashes is None:
        return None
    hashes = {n: v for n, v in hashes.items() if n in _SUPPORTED_HASHES}
    if not hashes:
        return None
    return hashes


def _clean_url_path_part(part: str) -> str:
    """
    Clean a "part" of a URL path (i.e. after splitting on "@" characters).
    """
    # We unquote prior to quoting to make sure nothing is double quoted.
    return urllib.parse.quote(urllib.parse.unquote(part))


def _clean_file_url_path(part: str) -> str:
    """
    Clean the first part of a URL path that corresponds to a local
    filesystem path (i.e. the first part after splitting on "@" characters).
    """
    # We unquote prior to quoting to make sure nothing is double quoted.
    # Also, on Windows the path part might contain a drive letter which
    # should not be quoted. On Linux where drive letters do not
    # exist, the colon should be quoted. We rely on urllib.request
    # to do the right thing here.
    return urllib.request.pathname2url(urllib.request.url2pathname(part))


