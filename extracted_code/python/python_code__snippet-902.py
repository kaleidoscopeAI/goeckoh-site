def sanitize_content_filename(filename: str) -> str:
    """
    Sanitize the "filename" value from a Content-Disposition header.
    """
    return os.path.basename(filename)


def parse_content_disposition(content_disposition: str, default_filename: str) -> str:
    """
    Parse the "filename" value from a Content-Disposition header, and
    return the default filename if the result is empty.
    """
    m = email.message.Message()
    m["content-type"] = content_disposition
    filename = m.get_param("filename")
    if filename:
        # We need to sanitize the filename to prevent directory traversal
        # in case the filename contains ".." path parts.
        filename = sanitize_content_filename(str(filename))
    return filename or default_filename


def _get_http_response_filename(resp: Response, link: Link) -> str:
    """Get an ideal filename from the given HTTP response, falling back to
    the link filename if not provided.
    """
    filename = link.filename  # fallback
    # Have a look at the Content-Disposition header for a better guess
    content_disposition = resp.headers.get("content-disposition")
    if content_disposition:
        filename = parse_content_disposition(content_disposition, filename)
    ext: Optional[str] = splitext(filename)[1]
    if not ext:
        ext = mimetypes.guess_extension(resp.headers.get("content-type", ""))
        if ext:
            filename += ext
    if not ext and link.url != resp.url:
        ext = os.path.splitext(resp.url)[1]
        if ext:
            filename += ext
    return filename


def _http_get_download(session: PipSession, link: Link) -> Response:
    target_url = link.url.split("#", 1)[0]
    resp = session.get(target_url, headers=HEADERS, stream=True)
    raise_for_status(resp)
    return resp


class Downloader:
    def __init__(
        self,
        session: PipSession,
        progress_bar: str,
    ) -> None:
        self._session = session
        self._progress_bar = progress_bar

    def __call__(self, link: Link, location: str) -> Tuple[str, str]:
        """Download the file given by link into location."""
        try:
            resp = _http_get_download(self._session, link)
        except NetworkConnectionError as e:
            assert e.response is not None
            logger.critical(
                "HTTP error %s while getting %s", e.response.status_code, link
            )
            raise

        filename = _get_http_response_filename(resp, link)
        filepath = os.path.join(location, filename)

        chunks = _prepare_download(resp, link, self._progress_bar)
        with open(filepath, "wb") as content_file:
            for chunk in chunks:
                content_file.write(chunk)
        content_type = resp.headers.get("Content-Type", "")
        return filepath, content_type


class BatchDownloader:
    def __init__(
        self,
        session: PipSession,
        progress_bar: str,
    ) -> None:
        self._session = session
        self._progress_bar = progress_bar

    def __call__(
        self, links: Iterable[Link], location: str
    ) -> Iterable[Tuple[Link, Tuple[str, str]]]:
        """Download the files given by links into location."""
        for link in links:
            try:
                resp = _http_get_download(self._session, link)
            except NetworkConnectionError as e:
                assert e.response is not None
                logger.critical(
                    "HTTP error %s while getting %s",
                    e.response.status_code,
                    link,
                )
                raise

            filename = _get_http_response_filename(resp, link)
            filepath = os.path.join(location, filename)

            chunks = _prepare_download(resp, link, self._progress_bar)
            with open(filepath, "wb") as content_file:
                for chunk in chunks:
                    content_file.write(chunk)
            content_type = resp.headers.get("Content-Type", "")
            yield link, (filepath, content_type)


