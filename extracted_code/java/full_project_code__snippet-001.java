"""
Checks whether the request of a response has been a HEAD-request.
Handles the quirks of AppEngine.

:param http.client.HTTPResponse response:
    Response to check if the originating request
    used 'HEAD' as a method.
"""
# FIXME: Can we do this somehow without accessing private httplib _method?
method = response._method
if isinstance(method, int):  # Platform-specific: Appengine
    return method == 3
return method.upper() == "HEAD"


