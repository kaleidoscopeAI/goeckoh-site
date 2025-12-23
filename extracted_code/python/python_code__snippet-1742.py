def warning(self, response: HTTPResponse) -> str | None:
    """
    Return a valid 1xx warning header value describing the cache
    adjustments.

    The response is provided too allow warnings like 113
    http://tools.ietf.org/html/rfc7234#section-5.5.4 where we need
    to explicitly say response is over 24 hours old.
    """
    return '110 - "Response is Stale"'

def update_headers(self, response: HTTPResponse) -> dict[str, str]:
    """Update the response headers with any new headers.

    NOTE: This SHOULD always include some Warning header to
          signify that the response was cached by the client, not
          by way of the provided headers.
    """
    return {}

def apply(self, response: HTTPResponse) -> HTTPResponse:
    updated_headers = self.update_headers(response)

    if updated_headers:
        response.headers.update(updated_headers)
        warning_header_value = self.warning(response)
        if warning_header_value is not None:
            response.headers.update({"Warning": warning_header_value})

    return response


