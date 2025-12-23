"""
Returns True if the connection is dropped and should be closed.

:param conn:
    :class:`http.client.HTTPConnection` object.

Note: For platforms like AppEngine, this will always return ``False`` to
let the platform handle connection recycling transparently for us.
"""
sock = getattr(conn, "sock", False)
if sock is False:  # Platform-specific: AppEngine
    return False
if sock is None:  # Connection already closed (such as by httplib).
    return True
try:
    # Returns True if readable, which here means it's been dropped
    return wait_for_read(sock, timeout=0.0)
except NoWayToWaitForSocketError:  # Platform-specific: AppEngine
    return False


