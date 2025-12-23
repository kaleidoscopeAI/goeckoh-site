def send(
    self,
    request: PreparedRequest,
    stream: bool = False,
    timeout: Optional[Union[float, Tuple[float, float]]] = None,
    verify: Union[bool, str] = True,
    cert: Optional[Union[str, Tuple[str, str]]] = None,
    proxies: Optional[Mapping[str, str]] = None,
) -> Response:
    pathname = url_to_path(request.url)

    resp = Response()
    resp.status_code = 200
    resp.url = request.url

    try:
        stats = os.stat(pathname)
    except OSError as exc:
        # format the exception raised as a io.BytesIO object,
        # to return a better error message:
        resp.status_code = 404
        resp.reason = type(exc).__name__
        resp.raw = io.BytesIO(f"{resp.reason}: {exc}".encode("utf8"))
    else:
        modified = email.utils.formatdate(stats.st_mtime, usegmt=True)
        content_type = mimetypes.guess_type(pathname)[0] or "text/plain"
        resp.headers = CaseInsensitiveDict(
            {
                "Content-Type": content_type,
                "Content-Length": stats.st_size,
                "Last-Modified": modified,
            }
        )

        resp.raw = open(pathname, "rb")
        resp.close = resp.raw.close

    return resp

def close(self) -> None:
    pass


