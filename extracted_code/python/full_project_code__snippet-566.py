"""ProxyManager does not support the supplied scheme"""

# TODO(t-8ch): Stop inheriting from AssertionError in v2.0.

def __init__(self, scheme):
    # 'localhost' is here because our URL parser parses
    # localhost:8080 -> scheme=localhost, remove if we fix this.
    if scheme == "localhost":
        scheme = None
    if scheme is None:
        message = "Proxy URL had no scheme, should start with http:// or https://"
    else:
        message = (
            "Proxy URL had unsupported scheme %s, should use http:// or https://"
            % scheme
        )
    super(ProxySchemeUnknown, self).__init__(message)


