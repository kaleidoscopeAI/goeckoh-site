"""Raised when the maximum number of retries is exceeded.

:param pool: The connection pool
:type pool: :class:`~urllib3.connectionpool.HTTPConnectionPool`
:param string url: The requested Url
:param exceptions.Exception reason: The underlying error

"""

def __init__(self, pool, url, reason=None):
    self.reason = reason

    message = "Max retries exceeded with url: %s (Caused by %r)" % (url, reason)

    RequestError.__init__(self, pool, url, message)


