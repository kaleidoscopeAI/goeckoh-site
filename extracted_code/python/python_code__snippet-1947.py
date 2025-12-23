"""
Return the host-port pair from a netloc.
"""
url = build_url_from_netloc(netloc)
parsed = urllib.parse.urlparse(url)
return parsed.hostname, parsed.port


