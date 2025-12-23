"""Raised when the connection to a proxy fails."""

def __init__(self, message, error, *args):
    super(ProxyError, self).__init__(message, error, *args)
    self.original_error = error


