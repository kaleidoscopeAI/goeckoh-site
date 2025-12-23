"""Raised when an existing pool gets a request for a foreign host."""

def __init__(self, pool, url, retries=3):
    message = "Tried to open a foreign host with url: %s" % url
    RequestError.__init__(self, pool, url, message)
    self.retries = retries


