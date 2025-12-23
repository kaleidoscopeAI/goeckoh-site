"""
A class to work around a bug in some Python 3.2.x releases.
"""
# There's a bug in the base version for some 3.2.x
# (e.g. 3.2.2 on Ubuntu Oneiric). If a Location header
# returns e.g. /abc, it bails because it says the scheme ''
# is bogus, when actually it should use the request's
# URL for the scheme. See Python issue #13696.
def http_error_302(self, req, fp, code, msg, headers):
    # Some servers (incorrectly) return multiple Location headers
    # (so probably same goes for URI).  Use first header.
    newurl = None
    for key in ('location', 'uri'):
        if key in headers:
            newurl = headers[key]
            break
    if newurl is None:  # pragma: no cover
        return
    urlparts = urlparse(newurl)
    if urlparts.scheme == '':
        newurl = urljoin(req.get_full_url(), newurl)
        if hasattr(headers, 'replace_header'):
            headers.replace_header(key, newurl)
        else:
            headers[key] = newurl
    return BaseRedirectHandler.http_error_302(self, req, fp, code, msg,
                                              headers)

http_error_301 = http_error_303 = http_error_307 = http_error_302


