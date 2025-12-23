"""
Cache the response by providing an expires 1 day in the
future.
"""

def update_headers(self, response: HTTPResponse) -> dict[str, str]:
    headers = {}

    if "expires" not in response.headers:
        date = parsedate(response.headers["date"])
        expires = expire_after(timedelta(days=1), date=datetime(*date[:6], tzinfo=timezone.utc))  # type: ignore[misc]
        headers["expires"] = datetime_to_header(expires)
        headers["cache-control"] = "public"
    return headers


