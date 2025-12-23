"""Transform and replace netloc in a url.

transform_netloc is a function taking the netloc and returning a
tuple. The first element of this tuple is the new netloc. The
entire tuple is returned.

Returns a tuple containing the transformed url as item 0 and the
original tuple returned by transform_netloc as item 1.
"""
purl = urllib.parse.urlsplit(url)
netloc_tuple = transform_netloc(purl.netloc)
# stripped url
url_pieces = (purl.scheme, netloc_tuple[0], purl.path, purl.query, purl.fragment)
surl = urllib.parse.urlunsplit(url_pieces)
return surl, cast("NetlocTuple", netloc_tuple)


