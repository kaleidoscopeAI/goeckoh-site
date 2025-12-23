def _get_http_response_size(resp: Response) -> Optional[int]:
    try:
        return int(resp.headers["content-length"])
    except (ValueError, KeyError, TypeError):
        return None


def _prepare_download(
    resp: Response,
    link: Link,
    progress_bar: str,
