"""
Cache **all** requests for a defined time period.
"""

def __init__(self, **kw: Any) -> None:
    self.delta = timedelta(**kw)

def update_headers(self, response: HTTPResponse) -> dict[str, str]:
    expires = expire_after(self.delta)
    return {"expires": datetime_to_header(expires), "cache-control": "public"}

def warning(self, response: HTTPResponse) -> str | None:
    tmpl = "110 - Automatically cached for %s. Response might be stale"
    return tmpl % self.delta


