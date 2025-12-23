def unpack_vcs_link(link: Link, location: str, verbosity: int) -> None:
    vcs_backend = vcs.get_backend_for_scheme(link.scheme)
    assert vcs_backend is not None
    vcs_backend.unpack(location, url=hide_url(link.url), verbosity=verbosity)


class File:
    def __init__(self, path: str, content_type: Optional[str]) -> None:
        self.path = path
        if content_type is None:
            self.content_type = mimetypes.guess_type(path)[0]
        else:
            self.content_type = content_type


def get_http_url(
    link: Link,
    download: Downloader,
    download_dir: Optional[str] = None,
    hashes: Optional[Hashes] = None,
